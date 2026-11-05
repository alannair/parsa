# Operating-point drift: shadow mode

Shadow mode counts offered demand and evaluates compatible catalog banks. A
`RETUNE` event is a **conditional recommendation**, never an executed change or
measured energy saving. The controller retains the existing presets, active
request profile table, routing policy, and backend configuration. Live mode is
unconditionally rejected, even with a populated catalog.

## Configuration

Pass the existing `routing_kwargs` object to `EnergyAwareController`, or serialize
that object into the existing CLI `--routing-kwargs` option:

```python
routing_kwargs = {
    "op_drift": {
        "mode": "shadow",
        "catalog_path": "/path/to/operator-supplied-catalog.json",
        "runtime_context": operator_declared_context,
    }
}
```

`operator_declared_context` must contain exactly these catalog compatibility keys:

| Field | Required value |
| --- | --- |
| `model`, `model_revision` | Nonempty identity strings |
| `versions` | Object with nonempty `vllm` and `vllm_ascend` version strings |
| `tp` | Positive integer tensor parallelism |
| `precision` | Nonempty string |
| `max_num_batched_tokens` | Positive integer fixed engine token budget (MBT) |
| `device_cohort` | Nonempty hardware/control-domain identity string |
| `serving_condition`, `serving_profile_hash` | Nonempty serving identity strings |
| `predictor_id`, `routing_policy` | Nonempty identity strings |
| `slo_definition` | Nonempty JSON identity object, with no null/boolean/empty identities |

The runtime context must canonically match the catalog compatibility object,
including numeric types. These are **operator declarations**, not model, predictor,
software, device, calibration, or deployment attestations. The controller can also
check the actual routing-policy name and the registry's lane IDs, frequencies,
engine concurrency caps, and fixed MBT against `initial_bank`. A mismatch fails
construction. Missing preset frequency/cap values are not wildcards. Both
`specialization_gap` and `op_conflict` reject non-off drift; the CLI rejects these
combinations before loading presets or building predictors.

Other existing `op_drift` options and defaults:

```json
{
  "window_s": 120.0,
  "evaluate_every_s": 30.0,
  "min_arrivals": 100,
  "max_observations": 10000,
  "persist_windows": 3,
  "min_residence_s": 600.0,
  "cooldown_s": 600.0,
  "benefit_horizon_s": 900.0,
  "min_saving_fraction": 0.05,
  "margin_j": 0.0
}
```

These are policy defaults, not experimentally calibrated production settings.
Times/counts must be positive and finite; counts must be integers. The saving
fraction is in `[0, 1)`; the energy margin is nonnegative joules. Unknown keys are
rejected. Off is the default: it does not read a catalog, create an observer/task,
emit drift events, or add `op_drift` to static config serialization.

## Actual catalog schema and units

The JSON loader accepts exactly these seven top-level fields:

* `schema_version`: the string `parsa-op-catalog-v1`.
* `compatibility`: the exact runtime identity object above.
* `feature_schema`: `revision="length-bins-v1"`, nonempty increasing positive
  `prompt_edges` and `output_edges` (token lengths), unique `slo_classes`, and
  matching `slo_targets` (success fractions in `(0, 1]`). Bins are half-open;
  equality belongs to the higher bin. Class IDs are `p<i>:o<j>:<slo>`; unavailable
  or invalid features use `unknown`.
* `operating_points`: OP-ID-to-object mapping, each with `frequency_mhz` (positive
  MHz), `max_num_seqs` (positive integer actual engine cap), and inline `profile`.
  `profile` contains exactly `a_ttft`, `b_ttft`, `a_tbt`, `b_tbt`, `a_ep`, `b_ep`,
  `a_ed`, `b_ed`, all finite. The existing linear convention is intercept plus
  slope times input-token count: TTFT and time-between-tokens are seconds, prefill
  energy is joules, and decode energy is joules per output token. These request
  profiles alone do not establish bank power/capacity envelopes; shadow does not
  bind them into the active serving profile.
* `initial_bank`: stable lane-ID-to-OP-ID mapping. Exactly one entry in `banks`
  must have this lane assignment.
* `banks`: bank-ID-to-object mapping with `lanes` and `regimes`. Every bank has
  exactly the initial lane IDs; duplicate lane assignment maps are rejected.
  Each regime contains exactly:
  * `regime_id`: unique within that bank;
  * `arrival_rps`: inclusive `{min, max}` requests/second region;
  * `class_mix`: feature-ID-to-`{min, max}` fraction region; omitted classes have
    zero allowed mass, and every declared minimum is enforced;
  * `power_w`: `{lo, hi}` total-bank watt bounds;
  * `capacity_rps_lo`: nonnegative sustainable requests/second lower bound;
  * `slo_success_lo`: success-fraction lower bounds for every SLO class;
  * `calibration_revision` and nonempty `source_run_ids` provenance.
* `transitions`: list of directed one-lane edge records with exactly
  `source_bank`, `target_bank`, `lane_id`, `duration_hi_s` (positive seconds),
  `extra_energy_hi_j` (nonnegative additional joules),
  `remaining_capacity_rps_lo` (positive requests/second while the lane is absent),
  `max_backlog_requests` (nonnegative integer), `calibration_revision`, and
  nonempty `source_run_ids`.

Duplicate JSON keys, nonfinite numbers, invalid references, and incompatible
identities are rejected. Canonical SHA-256 hashes identify content, not its
authenticity. Empty `regimes` and empty `transitions` are structurally valid;
after warmup/sample gates, absent measured coverage yields `NO_CHANGE` with
`reason="no_calibration"`. There is no extrapolation or online fitting. Fixtures
in `tests/test_op_drift.py` are synthetic **schema/arithmetic examples only**;
they are not usable measured calibration.

## Observation and headroom

At the normal router boundary, an arrival is counted before any admission/queue
await. Direct `choose`/`choose_or_wait` calls also count an arrival. A request-scoped
opaque context follows classification, queued work, dependency replans, the returned
Choice, and the terminal outcome. It is not stored in a controller ID map and is
never put into request bodies, headers, or metrics JSON. Observation consumes only
the existing `l_in`, **raw** predicted `l_out_raw`, and requested SLO class. The
existing dependency-only reclassification still runs; it cannot recount or rewrite
the original observation. Queue-full rejection before classification stays unknown.
The window retains only bounded numeric diagnostics and bucket counters, not prompts.

Collector, policy state, enrichment, completions, snapshots, and headroom collection
use one monotonic clock. Enrichment uses current call time while retaining the
context's original arrival bucket. Completion-time counters are separate from
offered arrivals. Terminal statuses are `completed`, `failed`, `rejected`, and
`cancelled`, recorded once via context idempotence. This integration always supplies
`slo_met=None`: HTTP 200 and total proxy elapsed time do not prove joint TTFT/TPOT
success. Cancellation and original routing errors propagate normally.

For streaming responses, shadow mode also owns an observation-only ASGI execution
guard after the route handler returns. A lazily defined `StreamingResponse` subclass
preserves the framework's Response identity. Response-start/send errors record
`failed`; cancellation in response execution and a normal pre-iteration disconnect
record `cancelled`. Successful stream exhaustion records `completed`, with the
same opaque context preventing duplicate terminal counts from the iterator and
response guard. Returning a response or receiving HTTP 200 alone is not completion.
The guard only calls the observation hook; proxy release, lane-state cleanup and
reservation ownership remain with their existing paths. Off uses the original
`StreamingResponse` class and execution behavior.

Production headroom requires KV polling to be enabled and **every** catalog lane
to have a fresh matching-engine snapshot through `KVProjectionStore.get_snapshot`.
The store validates epoch-based freshness; the returned collection is then stamped
in the observer's monotonic domain. Future timestamps, invalid counters/capacity,
host swapping, missing lane accounting, or unsupported record states abstain.
NPU capacity must be a positive integer, and nonnegative integer used/free block
counts must sum **exactly** to capacity; both under-total and over-total readings
are invalid, consistent with the backend producer's `used = capacity - free`.
Disabled/missing/stale KV or unknown backlog returns no headroom; it never becomes
zero backlog or a fabricated healthy lane list. Compatible active registry values
are rechecked when collecting production headroom.

Backlog is calculated from existing sources:

```text
unbound = union of queued pending/blocked request IDs
          and admission-record pending/blocked request IDs
backlog = len(unbound)
          + sum over lanes (
              planned/released record count
              + max(running record count, gateway in_flight, backend active_requests)
            )
```

Bound planned/running records remove the same request from `unbound`. The union is
temporary per collection, not an observer request history. Running sources overlap;
planned work is additional because it may not yet have reached a backend. This can
overcount recently delivered plans and lagging snapshots. Backend `active_requests`
is the existing KV contract, not independent proof of all engine unfinished work
or exclusive gateway ownership. This is conditional shadow headroom, **not live
drain proof**. Existing queue/reservation cancellation behavior is not repaired by
observation; for example, a cancelled queued entry can remain in the existing queue
and be conservatively counted later.

Elapsed bucket exposure, cold-start/sample gates, empirical peak-bucket demand
bounds, one-winner persistence across distinct windows, residence, and cooldown are
the Task 2 policy. Recommendations assume that the recent empirical demand bound
persists through the calibrated transition; they do not establish future demand.

## Events and lifecycle

Read `op_drift` JSON lines from the existing `src.router_app` logger (INFO), or
filter `controller.metrics.snapshot()["events"]` for `event == "op_drift"` in
embedded use. There is no new metrics/actuation HTTP endpoint. Events share the
existing capped deque (default 512 events across all types), not a separate
unbounded decision history.

Each event has `ts` (epoch log time), `event="op_drift"`, `mode="shadow"`,
`executed=false`, `identity_source="operator_declared"`, `runtime_context`,
`catalog_hash`, `config_hash`, `observation`, `estimates`, and `decision`.
`observation` is the Task 2 numeric window snapshot, optionally with external
`headroom={fresh, at_s, healthy_lane_ids, backlog_requests}`. `estimates` maps bank
IDs to the matched conservative evidence or null; evidence includes power/capacity/
SLO bounds, coverage IDs, regime IDs and calibration/source provenance.
`decision` is the selector result **without `policy_state`**, including its reason,
current/target bank, assessed lane, `expected_generation`, observation/coverage IDs,
benefit/cost/saving bounds and projected transition backlog. The initial generation
zero is only a static shadow label, not a verified backend epoch or committed
generation. Only `action="RETUNE"` recommends a candidate; targets on abstentions
can identify an assessed but rejected candidate.

Errors emit `decision={action:"NO_CHANGE", reason:"shadow_error", current_bank,
target_bank}` and `stage`/`error_type`, with null observation/estimates. Exception
messages and request bodies are excluded. Errors clear pending-winner persistence
and the previous observation. A failed event sink attempts a fixed, data-free
warning; failure of that sink is also isolated. Observation may be incomplete when
its collection/recording fails; diagnostics are not a substitute for measurements.

`controller.shadow.snapshot()` and `controller.shadow.tick(headroom=None)` are
synchronous embedding interfaces; tick returns a serializable decision and retains
the returned selector state privately. `await controller.shadow.run(producer)`
evaluates immediately, then sleeps `evaluate_every_s` between attempts, including
errors. `producer` is an async headroom callable. `shadow_lifespan(controller)`
starts exactly one task when enabled, cancels it, and awaits it on exit.
`create_app` uses that same context manager independently of its KV/planner loops.
Web dependencies remain lazily imported. Do not call manual ticks concurrently
with the slow loop or from another thread; the observer has one event-loop owner.

Shadow performance results must not be pooled with off-mode baseline results
without accounting for the added observation/evaluation overhead. `serving_condition`
and static result formats retain their existing meanings; no adaptive execution,
bank history, or measured energy improvement is claimed.

## Offline verification

```sh
ENERGY_AWARE_SEQ_PRED_BERT_PKL=__parsa_offline_missing_bundle__.pkl python3 -m unittest tests.test_op_drift tests.test_router_modes tests.test_soft_admission -v
```

Pure controller, selector, router-object, and standalone async lifecycle tests need
only the standard-library test runner. In the current dependency-free environment,
the HTTP/FastAPI lifecycle integration test explicitly skips because aiohttp,
FastAPI, Starlette, httpx, and uvicorn are absent. No server or hardware behavior is
validated by those pure tests.
Streaming ASGI hand-off regressions use minimal standard-library Response bases
temporarily patched into `sys.modules` and a synthetic upstream. They exercise
execution after the real router handler returns, including before the first
iterator advance; they do not constitute real HTTP/framework integration coverage.
