# Stateful pivot verifier

A pivot is one rejected assistant turn after a verified public dialogue prefix.
Training generates one replacement turn. The verifier restores the certified
boundary and returns binary reward, without a model or a judge call. A passing
turn need not finish the dialogue.

## Data and code

Use the accepted `final-20261018/` bundle:

```
/lustre/fsw/portfolios/nemotron/projects/nemotron_speech_chat/users/deshr/results/tau-voice-rl/pivot-dataset/final-20261018/
  areal/{routes.json,records.jsonl,areal_consent/sample.jsonl,areal_write/sample.jsonl}
  afterquery/{routes.json,records.jsonl,afterquery_write/sample.jsonl}
  verifier/{FIELD_TABLE.json,CACHE_MANIFEST.json,V21_CACHE_MANIFEST.json,
            STAGING.json,stage.py,check_controls.py,services.yaml}
  DATACARD.md
```

There are 7,625 AReaL v1.1 and 294 AfterQuery v2.1-S1 records. Route paths are
relative to their dataset directory. Samples contain system + full public
prefix + tools, `agent_ref=tau_pivot_stateful_agent`, and `verification_ref`.
Private failure evidence and two controls live in `records.jsonl`, keyed by
`record_id`; never append them to a policy prompt. `s1_kind` describes the
AfterQuery selection; `read_pass_possible` flags the I35 limitation below.
See the datacard and SHA256SUMS for exact counts and pins.

## Stage and start (Linux x86-64, CPU)

Install the vendored Gym dependencies in this checkout (Python 3.13). No model
weights are needed to serve or check rewards. Use a node-local disk with at
least 160 GiB free; archive paths and exact compressed/extracted sizes are in
`verifier/STAGING.json`. The archives contain the pinned Python/native
checkers. Do not import them directly from Lustre or rebuild them.

```bash
export PIVOT_BUNDLE=/path/to/final-20261018
export PIVOT_STAGE=/tmp/pivot-final
export PIVOT_CODE=/path/to/this/checkout
python "$PIVOT_BUNDLE/verifier/stage.py" \
  --bundle "$PIVOT_BUNDLE" --stage "$PIVOT_STAGE" --code "$PIVOT_CODE"
export PYTHONPATH="$PIVOT_CODE/3rdparty/Gym-workspace/Gym"
python "$PIVOT_BUNDLE/verifier/check_controls.py" \
  --bundle "$PIVOT_BUNDLE" --stage "$PIVOT_STAGE" --serve \
  --out /tmp/pivot-controls
```

`stage.py` is stdlib-only: copy or download the pinned archives, SHA-check them,
extract snapshots and runtime bytes, and render manifests plus `services.json`
from the shipped services template. `--archive-root` optionally supplies a
local mirror of the archive paths in STAGING.json. Source archives default to
canonical mounted paths; `--ssh-host iad-login` downloads absent ones. The
script verifies this checkout against the bundle's exact verifier code pins.
It refuses to reuse an existing stage (use a new directory for a new version).
No cache capture or checker mutation occurs.

`check_controls.py --serve` starts only the resources server on loopback using
the staged services config, then sends all 15,838 controls through real HTTP
`/verify`, writes incremental receipts, and stops it. Omit `--serve` and provide
`--url http://HOST:PORT` to check an already running resources server.
To serve continuously use `check_controls.py --serve-only --port 8080` with the
same `--bundle` and `--stage`; this does not start a policy/model server.

For a Gym training deployment, compose
`voicechat/configs/gym-services-pivot-stateful.yaml` and set `pivot_bundle` and
`pivot_stage` to these paths. The services file wires the existing one-turn
`tool_simulation_agent` to this resource server and your `policy_model`.
The policy API key/base URL/model name are caller-provided for generation;
they are unused by the control check. There is no agent modification.

## Configuration and request

`cache_manifest` selects AReaL v1 snapshots; `runtime_manifest` defines pinned
native subprocesses. `v21_cache_manifest` and `v21_field_table` select the
AfterQuery namespace and frozen canonicalization table. `dispatch_policy` is
`affinity_steal`; eight runtime pools have four replicas each (32 workers),
`cache_entries=2` bounds loaded boundary states per worker, and
`worker_timeout_seconds=300` bounds a native request. Round-robin and fixed
record affinity are also supported. A worker exit returns reward 0 with
`worker_exited` diagnostics and respawns the worker; treat diagnostics errors
as infrastructure failures, not valid negative training examples.

POST `/verify` with the sample's `responses_create_params`, its complete
`verification_ref` (`record_id`, `dataset`, `verifier_version`,
`expected_turn_kind`), and the generated OpenAI Responses API `response`.
The control-check script supplies a concrete request example. The response
includes `reward` and `diagnostics.predicates`.

AReaL v1 runs the frozen archive-specific native step checks plus the boundary
pivot predicates. Consent pivots require a message satisfying the content
predicates; write pivots preview the action on the restored state. AfterQuery
v2.1 matches one assistant tool call against the remaining public gold ledger,
using the frozen schema field table: structural/enum/numeric/boolean/identifier
canonicalization and required content tokens for free text. All predicates
must pass to earn 1. The recorded rejected output scores 0 and the shipped
positive control scores 1. Exact code, table, states and references determine
the reward; zero judge/model calls and no training are needed for verification.

## Limitations

- AfterQuery description-only defaults can differ from human expectations;
  S1's independent audit was 27/30, not a guarantee of perfect labels.
- I35: a remaining read action can earn reward 1 without advancing a write.
  `read_pass_possible` records whether a remaining gold read actually passes
  at this pivot. This annotation does not change the reward.
- Telecom caller-action holds were excluded by certification; this is an
  assistant-turn dataset, not a caller recovery dataset.
- Full prefixes are retained, including 115 prompts above 14,336 rendered
  tokens and the approximately 153k-token maximum. No length filtering or
  model capacity assumption is imposed by this bundle.
- Private pickle snapshots are trusted pinned artifacts; do not load arbitrary
  untrusted cache files. Native state and reference-ledger matching measure a
  local one-turn condition, not full-dialogue completion or semantic perfection.
