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


## G5 development branch (not a certified dataset)

`desh/areal-pivot-v2` adds the `areal-v2:` evaluator namespace. Existing v1 and
AfterQuery v2.1 routes retain their original evaluators. The new namespace
requires replay-certified eligibility evidence, a boundary-state hash, a native
feasibility callback, and (for K4) a separately pinned relaxed consent callback.
It cannot consume the old consent cache callbacks as new eligibility evidence.

The evaluator implements typed gold-tool controls for K1/K2/K5/K6/K7, a hard
valid-prior-consent exclusion for K4, and a conservative grounded-refusal check
for K3. K3 alternatives and K7 continuation without an executable gold step
remain unsupported. K2 mining must exclude open-string-only differences under
S1; exact typed matching in the evaluator is not evidence that a mismatch is
a genuine error. The online SDG rejection label is never an eligibility proof.

The draft has mutation tests; full native mining, HTTP control parity and the
strategist's per-kind audit remain required before any v2 training publication.
There is no approved `v2-areal` bundle yet.


### G5 INBOX31 execution checkpoint (turn47)

The `desh/areal-pivot-v2` branch remains a draft, with no replacement training
bundle. `reference/recipes/tau_voice_rl_pivot/consent_given.py` decides actual
prior consent independently of SDG content grades. Identifying cues include public native
tool descriptions and native posting amounts, with exact decimal boundaries.
A posting amount is a fact, never an SDG C3 consent decision. K4 rechecks the fixed public
context during scoring; a stale false label cannot authorize a re-ask after yes.
K5 requires an actual question/request, not a declarative "To confirm" readback.

`areal_native.Boundary` retains all task actions, checks the source DB hash,
replays native tool results exactly, and proposes boundaries independently of
online rejection reasons. It holds unsupported actions and alternatives. The
seeded B0 pilot additionally requires raw/T20 caller/action/result identity;
matching task ID and seed alone is insufficient.

`pinned_consent.ConsentContent` grades proposed K4 messages using the approved
whole closure in INBOX31. The question's hypothetical yes is used only for
content grading. Checker files are supplied as node-local paths plus exact SHA
pins; they must be staged with any eventual bundle. Original bytes are retained.
Telecom uses its frozen C3 content rule and separately reports C1/C6 native
preconditions. An old template containing "modify or exchange" fails retail's
choice check; controls now name only the applicable action.

K3 is **not eligible for production certification**: the turn47 adversarial
battery found five false accepts in the draft lexical scorer. Full mining,
K1/K2/K3/K6/K7 native/control coverage, balanced selection, per-kind signal,
and the final bundle remain pending. Pilot and hard-gate receipts in the worklog
are narrower than these acceptance gates.

### G5 INBOX33 checkpoint (turn49)

The independent consent detector now has all 60 Fable turn48 labels as tests.
It supports affirmative action restatements, detail-grounded offers, explicit
caller go-ahead and separately scoped add-ons. Information answers after earlier
assent carry an uncertainty hold; they cannot earn a K4-positive re-ask. The
pinned content graders remain unchanged. K3 is deferred and always scores zero.

Full native mining is a separate frozen CPU job, not an accepted dataset. B0
prompts use T20 caller text; a tool-call or result byte difference remains a
source hold. Original v1 data and the shared verifier branch are unchanged.


G5 draft INBOX34: K4 requires no caller assent in the write segment. The
independent detector emits `consent_label`, `k4_eligible`, and the indexed
`write_segment` assent/ignored evidence. `valid_prior_consent` retains the
turn49 witness result; `authorized_prior_consent` additionally excludes
question/hold cases. Newly recovered census-given cases block K4 without
automatically creating K5 positives. No replacement G5 bundle is certified.

G5 INBOX35 draft: native callbacks expose the next ordered remaining step.
Reference read/auth calls absent from task evaluation criteria retain explicit
`recorded_reference_read` provenance. K4 requires `write_due`; K5 also needs a
question-offer witness and clean yes. Wrong/missing lookups and premature closes
use the due read as their target. A missing reference anchor fails closed. New
candidate states and controls must be replayed under this version before audit;
no G5 training bundle is accepted yet.

### G5 INBOX36 recovery R1 (candidate contract)

New mining explicitly constructs `Boundary(..., recovery_r1=True)` and specs
with `contract: G5-INBOX36-R1`. Earlier cached/spec versions keep their prior
behavior. Remaining writes are matched as a set; native execution and remaining
completion establish readiness in the actual state. A K6 write requires no exact
remaining match, unavailable public arguments, and a public native lookup that
supplies the intended target. K1 has no remaining matching write tool; K2 has
matching tool/target, wrong arguments, and a native consequence.

K5 additionally requires the offer to identify the exact order/reservation or
explicitly the profile/default address, with every gold argument public. K7 is
intentionally lenient: one nonempty spoken message or a call equal to any remaining
gold step, with no end/transfer. It does not validate spoken factual content.

B0 identifier-misrecognition K6 records carry `audio_only=true`; the default text
pool excludes them. Keep inclusive and text counts separately. INBOX36 suspends
kind-cap/training selection; audit candidates are not a released training bundle.
New native/HTTP qualification and strategist audit are required before publication.

### Released AReaL v2 pool (2026-10-23)

The `v2-areal-pool-20261023` bundle supplies all 1,848 eligible text pivots and
270 optional audio-only pivots. `MANIFEST.jsonl` labels K1/K2/K4/K5/K6/K7 and
marks the default diversity subset; no category mixture is selected. K3 remains
excluded. Read the bundle's verbatim `CATEGORIES.md` for reward definitions and
limitations. R2.1 specs opt in with `contract: G5-INBOX38-R21`; v1 semantics and
the frozen content-checker bytes remain unchanged.

On an allocated CPU node, stage the shipped code, runtimes and cached boundary
payloads onto local disk. No model, judge, state replay or GPU is required:

```bash
BUNDLE=/absolute/path/to/v2-areal-pool-20261023
STAGE=/tmp/pivot-v2-stage
python3 "$BUNDLE/verifier/stage.py" --bundle "$BUNDLE" --stage "$STAGE"
python3 "$BUNDLE/verifier/launch.py" --stage "$STAGE" \
  --serve --out /absolute/path/to/fresh-control-receipts
# Alternatively, keep the resources server available for policy clients:
python3 "$BUNDLE/verifier/launch.py" --stage "$STAGE" \
  --serve-only --port 8000
```

The all-runtime stage requires at least 160 GiB free local disk. For a smaller
control check, repeat `--runtime-key KEY` when staging; keys are listed in
`verifier/STAGING.json`. The generated `services.json`, `RUNTIME.json` and
`CACHE_MANIFEST.json` refer only to that local stage. Bindings, code and payload
hashes are checked before use. Use the trusted supplied snapshots only; they
contain executable Python serialization.

`POST /verify` takes `responses_create_params` and `verification_ref` from the
selected Gym sample plus the generated OpenAI Responses `response`. It returns
`reward` and `diagnostics`. The reference selects the cached state and kind;
private gold actions, evidence and controls belong in `records.jsonl`, not the
policy prompt. K4 uses the frozen content grader; K1/K2/K5 require the confirmed
write, K6 the missing native lookup/auth, and K7 a nonclosing message or an
allowed remaining step. K7 spoken factual correctness is not graded. The bundle
check runs every named control, including multiple positives, through this HTTP
path using the published public prompt. Check its final matched/expected counts
and diagnostics rather than accepting a partial progress line.
