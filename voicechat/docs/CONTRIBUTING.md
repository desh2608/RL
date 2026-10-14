# Sending findings back

Maintained by cchen1 (gitlab-master `cchen1/nemo-rl-voicechat`). Fixes and findings are welcome.

1. Branch from `main` as `<name>/<topic>`.
2. Code changes follow "Rules for code changes" in `voicechat/AGENTS.md`:
   - defaults keep the old behaviour, and the new behaviour sits behind a config key;
   - add a CPU-runnable test (`tests/unit/...` or `voicechat/tools/audit/`);
   - check it on a smoke with `smoke_check.py`.
3. Add an entry to `voicechat/docs/BUGS_FIXED.md` with the next number:
   - **symptom**: the metric or log line, with numbers;
   - **cause**: the code location;
   - **fix**: the file changed, or the config key that enables it;
   - **how to check**: the tool and the expected value;
   - whether it applies to offline, online or both.
4. Findings that are not fixed yet go in the same file, marked "risk" or "open", with a way to verify them.
5. Open a Merge Request with the above and the measured results. Run paths on cs-oci are fine, since others can read them.

**Do not commit**:
- checkpoints, data, run directories or logs;
- files over 3 MB;
- secrets.

Keep data and experiment artifacts on the cluster and link their paths in the MR.

**Third-party changes**: Gym is in the repo, so edit it directly. For Megatron-Bridge / Megatron-LM, `git format-patch`
in the respective tree, add the patch to `voicechat/third_party/*-patches/`, and update the base notes in
`fetch_third_party.sh`.
