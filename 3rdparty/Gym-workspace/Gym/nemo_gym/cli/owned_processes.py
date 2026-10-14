"""Bounded cleanup of Gym-owned sessions and observed descendants only."""
import ctypes
import os
import sys
import signal
from threading import Event, RLock, Thread
from time import monotonic

import psutil


class OwnedProcessRegistry:
    """Observe descendants before shell roots exit, including children that call setsid.

    Linux service process groups are created explicitly by RunHelper. No matching
    by executable name, environment, or global kill is used. PID creation times
    protect individually signalled descendants from PID reuse.
    """

    def __init__(self):
        self._old_subreaper = None
        if sys.platform == "linux":
            libc = ctypes.CDLL(None, use_errno=True)
            old = ctypes.c_int()
            if libc.prctl(37, ctypes.byref(old), 0, 0, 0) != 0 or libc.prctl(36, 1, 0, 0, 0) != 0:
                raise RuntimeError("Cannot establish Gym-owned orphan reaping")
            self._old_subreaper = old.value
        self._baseline_children = {}
        for child in psutil.Process().children():
            try:
                self._baseline_children[child.pid] = child.create_time()
            except psutil.NoSuchProcess:
                pass
        self._registration_failures = []
        self._lock = RLock()
        self._stop = Event()
        self._roots = {}
        self._known = {}
        self._errors = []
        self._refresh_stats = {"calls": 0, "total_seconds": 0.0, "max_seconds": 0.0, "identity_skips": 0}
        self._thread = Thread(target=self._observe, daemon=True)
        self._thread.start()

    def register(self, name, process):
        try:
            p = psutil.Process(process.pid)
            identity = self._identity(p, name)
        except psutil.NoSuchProcess:
            self._registration_failures.append({"server": name, "pid": process.pid,
                                                "reason": "root_already_reaped"})
            raise RuntimeError(f"Gym server {name} exited before ownership registration")
        if os.getpgid(p.pid) != p.pid or os.getsid(p.pid) != p.pid:
            raise RuntimeError(f"Gym server {name} does not own its process session")
        with self._lock:
            self._roots[name] = (process, identity)
            self._known[p.pid] = identity
            self._refresh()

    @staticmethod
    def _identity(process, name):
        return {"pid": process.pid, "create_time": process.create_time(), "server": name}

    @staticmethod
    def _same(identity):
        try:
            p = psutil.Process(identity["pid"])
            return p if p.create_time() == identity["create_time"] else None
        except psutil.NoSuchProcess:
            return None

    def _group_if_same(self, identity):
        p = self._same(identity)
        if p is None:
            return None
        try:
            return os.getpgid(p.pid)
        except ProcessLookupError:
            return None

    def _refresh(self):
        # One PPID table per sweep, rather than a full recursive process scan for
        # each known descendant. Only identities reached from owned roots/known
        # descendants are retained; no selection by name or user is performed.
        started = monotonic()
        table = {}
        children = {}
        for process in psutil.process_iter(attrs=["pid", "ppid", "create_time"], ad_value=None):
            row = process.info
            if row["ppid"] is None or row["create_time"] is None:
                continue
            table[row["pid"]] = row
            children.setdefault(row["ppid"], []).append(row["pid"])
        pending = [identity for identity in self._known.values()
                   if table.get(identity["pid"], {}).get("create_time") == identity["create_time"]
                   and self._same(identity) is not None]
        visited = set()
        while pending:
            parent = pending.pop()
            if parent["pid"] in visited or self._same(parent) is None:
                continue
            visited.add(parent["pid"])
            for pid in children.get(parent["pid"], []):
                row = table[pid]
                try:
                    # process_iter caches Process objects; fresh identity and PPID
                    # checks prevent cached births from attributing a reused PID.
                    child = psutil.Process(pid)
                    if child.create_time() != row["create_time"] or child.ppid() != parent["pid"] or self._same(parent) is None:
                        self._refresh_stats["identity_skips"] += 1
                        continue
                except psutil.NoSuchProcess:
                    self._refresh_stats["identity_skips"] += 1
                    continue
                identity = {"pid": pid, "create_time": row["create_time"], "server": parent["server"]}
                self._known[pid] = identity
                pending.append(identity)
        elapsed = monotonic() - started
        self._refresh_stats["calls"] += 1
        self._refresh_stats["total_seconds"] += elapsed
        self._refresh_stats["max_seconds"] = max(self._refresh_stats["max_seconds"], elapsed)

    def _observe(self):
        while not self._stop.wait(0.2):
            try:
                with self._lock:
                    self._refresh()
            except Exception as exc:
                with self._lock:
                    self._errors.append(type(exc).__name__)
                return

    def _reap(self):
        for process, _ in self._roots.values():
            process.poll()  # Popen owns and reaps its direct child.
        root_pids = {process.pid for process, _ in self._roots.values()}
        for identity in self._known.values():
            if identity["pid"] in root_pids or self._same(identity) is None:
                continue
            try:
                os.waitpid(identity["pid"], os.WNOHANG)
            except (ChildProcessError, ProcessLookupError):
                pass  # Still owned by a live service parent, or already reaped.

    def _remaining(self):
        result = []
        for identity in self._known.values():
            p = self._same(identity)
            if p is not None:
                try:
                    result.append(dict(identity, status=p.status(), pgid=os.getpgid(p.pid)))
                except psutil.NoSuchProcess:
                    pass
                except ProcessLookupError:
                    pass
        return result

    def shutdown(self, graceful_seconds=5, terminate_seconds=3, kill_seconds=3):
        phases = []
        escaped = {}
        try:
            with self._lock:
                self._refresh()
            for sig, seconds in ((signal.SIGINT, graceful_seconds),
                                 (signal.SIGTERM, terminate_seconds),
                                 (signal.SIGKILL, kill_seconds)):
                with self._lock:
                    self._refresh()
                    self._reap()
                    remaining = self._remaining()
                    if not remaining:
                        break
                    groups = {identity["pid"] for _, identity in self._roots.values()}
                    live_groups = {item["pgid"] for item in remaining
                                   if item["pgid"] in groups}
                    for pgid in live_groups:
                        # At least one PID/create_time-matched owned member must
                        # still occupy this group; never signal empty historical IDs.
                        if not any(self._group_if_same(item) == pgid
                                   for item in remaining if item["pgid"] == pgid):
                            continue
                        try:
                            os.killpg(pgid, sig)
                        except ProcessLookupError:
                            pass
                    for identity in remaining:
                        if identity["pgid"] in groups or identity["status"] == psutil.STATUS_ZOMBIE:
                            continue
                        escaped[identity["pid"]] = identity
                        p = self._same(identity)
                        if p is not None:
                            try:
                                p.send_signal(sig)
                            except psutil.NoSuchProcess:
                                pass
                    phases.append({"signal": sig.name, "processes_before": len(remaining)})
                deadline = monotonic() + seconds
                while monotonic() < deadline:
                    with self._lock:
                        self._refresh()
                        self._reap()
                        if not self._remaining():
                            break
                    self._stop.wait(0.05)
            with self._lock:
                self._refresh()
                self._reap()
                remaining = self._remaining()
                unattributed = []
                for p in psutil.Process().children():
                    try:
                        if p.pid not in self._known and self._baseline_children.get(p.pid) != p.create_time():
                            unattributed.append({"pid": p.pid, "create_time": p.create_time(), "status": p.status()})
                    except psutil.NoSuchProcess:
                        pass
                report = {"status": "PASS" if not remaining and not self._errors and not unattributed and not self._registration_failures else "FAIL",
                          "registered": list(self._known.values()),
                          "escaped_processes": list(escaped.values()),
                          "remaining": remaining, "observer_errors": list(self._errors),
                          "unattributed_children": unattributed,
                          "registration_failures": self._registration_failures,
                          "subreaper_before": self._old_subreaper,
                          "refresh_stats": dict(self._refresh_stats),
                          "phases": phases}
        finally:
            self._stop.set()
            self._thread.join(timeout=2)
            if self._old_subreaper is not None:
                if ctypes.CDLL(None, use_errno=True).prctl(36, self._old_subreaper, 0, 0, 0) != 0:
                    raise RuntimeError("Cannot restore prior subreaper setting")
        if self._thread.is_alive():
            raise RuntimeError("Gym descendant observer did not exit")
        return report
