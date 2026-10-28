# Reproduce the combined pivot-pool selection

These are the audited INBOX69/71 admission, dedupe and limit modules. The eight
files in AUDITED_SOURCES.json are byte-identical to the accepted turn88 sources.
Some helpers retain historical builders for provenance; use the entry point below.
No native environment, HTTP verifier, model or judge is called.

On an IAD CPU allocation, copy this directory to node-local disk and run:

```bash
BUNDLE=/absolute/path/to/v4-combined-pool-20261028
cp -r /path/to/RL/tools/pivot_pool /tmp/pivot-pool-code
python3 -I /tmp/pivot-pool-code/test_select_pool.py
python3 -I /tmp/pivot-pool-code/reproduce_selection.py \
  --bundle "$BUNDLE" --out /path/to/new/selection-reproduction
```

The command stages the bundle's pinned selection_reference under /tmp, recomputes
H1prime/H2/H3/H4prime, uses only source-eligible text records, then applies v4>v3>v2
priority, global task/category/prefix dedupe, within-dialogue category/state dedupe,
3/dialogue, 3/task/category, 50/scenario and the 30% category cap (35/40 variants
also reproduced). It checks every record's three selection fields and compares
hold decisions, duplicate drops, near counts, limit drops and capacities with the
bundle. SELECTION_REPRODUCTION.json reports PASS only after all comparisons agree.
The accepted bundle has 3,639 default text records and 270 separate audio records.

The frozen turn86 inventory is additionally required to rerun the labelled
143-record battery and old-kept H4 regression set:

```bash
# Use the same node-local code and stage selection_reference alongside it first.
cp -r "$BUNDLE/verifier/selection_reference" /tmp/selection_reference
PIVOT_HOLD_REFERENCE=/tmp/selection_reference \
python3 -I /tmp/pivot-pool-code/test_battery71.py \
  --source /absolute/path/to/v4-combined-pool-20261028-turn86 --out /tmp/BATTERY.json
```

That battery includes all 135 old H1 cases, eight labelled H4 cases, ten previously
kept H4 samples, four unlabelled decisions and a security-enrollment scope guard.
H2/H3 evidence must match on all 6,300 records. Source eligibility is historical;
HOLD_DECISIONS.jsonl supplies extra admission vetoes. No gold action or private
boundary state belongs in the policy prompt. See the bundle README for scoring,
verifier staging, source provenance, count tables and accepted limitations.
