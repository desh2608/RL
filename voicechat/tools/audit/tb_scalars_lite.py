#!/usr/bin/env python3
"""Read scalar tags from TensorBoard tfevents files with no tensorboard/protobuf dependency.

Usage: tb_scalars_lite.py <tfevents file or dir> --tags REGEX [--table]
Prints JSON lines {tag, step, value, wall_time} or, with --table, one row per step.
Only `simple_value` scalars (what torch SummaryWriter.add_scalar writes) are decoded.
"""
import argparse, json, os, re, struct, sys
from collections import defaultdict


def _varint(buf, i):
    shift = result = 0
    while True:
        b = buf[i]; i += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, i
        shift += 7


def _fields(buf):
    """Yield (field_number, wire_type, value) for one protobuf message."""
    i, n = 0, len(buf)
    while i < n:
        key, i = _varint(buf, i)
        fn, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(buf, i)
        elif wt == 1:
            v = buf[i:i + 8]; i += 8
        elif wt == 2:
            ln, i = _varint(buf, i); v = buf[i:i + ln]; i += ln
        elif wt == 5:
            v = buf[i:i + 4]; i += 4
        else:
            raise ValueError(f"unsupported wire type {wt}")
        yield fn, wt, v


def records(path):
    with open(path, 'rb') as f:
        while True:
            head = f.read(8)
            if len(head) < 8:
                return
            (ln,) = struct.unpack('<Q', head)
            f.read(4)
            data = f.read(ln)
            f.read(4)
            if len(data) < ln:
                return
            yield data


def scalars(path):
    for rec in records(path):
        wall = step = None; summary = None
        for fn, wt, v in _fields(rec):
            if fn == 1 and wt == 1: wall = struct.unpack('<d', v)[0]
            elif fn == 2 and wt == 0: step = v
            elif fn == 5 and wt == 2: summary = v
        if summary is None:
            continue
        for fn, wt, val in _fields(summary):
            if fn != 1 or wt != 2:
                continue
            tag = value = None
            for f2, w2, v2 in _fields(val):
                if f2 == 1 and w2 == 2: tag = v2.decode('utf-8', 'replace')
                elif f2 == 2 and w2 == 5: value = struct.unpack('<f', v2)[0]
            if tag is not None and value is not None:
                yield {'tag': tag, 'step': step, 'value': value, 'wall_time': wall}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path'); ap.add_argument('--tags', default='.'); ap.add_argument('--table', action='store_true')
    a = ap.parse_args()
    files = [a.path] if os.path.isfile(a.path) else sorted(
        os.path.join(r, f) for r, _, fs in os.walk(a.path) for f in fs if 'tfevents' in f)
    pat = re.compile(a.tags)
    rows = [s for f in files for s in scalars(f) if pat.search(s['tag'])]
    if not a.table:
        for s in rows: print(json.dumps(s))
        return
    by = defaultdict(dict); tags = sorted({s['tag'] for s in rows})
    for s in rows: by[s['step']][s['tag']] = s['value']
    print('step\t' + '\t'.join(tags))
    for step in sorted(by):
        print(f"{step}\t" + '\t'.join(f"{by[step].get(t, float('nan')):.4f}" for t in tags))


if __name__ == '__main__':
    main()
