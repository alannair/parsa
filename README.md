# PARSA

## Deployment

Run preparation and serving commands from the repository root.
The explicit recipes are `setup/Dockerfile.0.11` and `setup/Dockerfile.0.23`; run image-build commands from `setup/`, which remains the Docker build context. Each Dockerfile lists its own application dependencies. There is no generic `setup/Dockerfile`; from `setup/`, manual builds must select `-f Dockerfile.0.11` or `-f Dockerfile.0.23`.
The host needs Docker with Buildx and compatible Ascend devices/driver, with the selected model under `/data1/models` (default: `qwen3-4b`).
The container mounts this repository at `/workspace`, models at `/models`, and `DATASET_DIR` (default: `/data1/datasets`) at `/datasets`.

### v0.11 legacy default

Omitting `--version` selects v0.11 for every action; explicit `--version 0.11` selects the same workflow.

```bash
python3 setup/deploy.py prepare-vllm
(cd setup && python3 deploy.py build-image parsa)
python3 setup/deploy.py run parsa --devices 0,1
```

This uses `quay.io/ascend/vllm-ascend:v0.11.0`, prepares `setup/vllm-upstream`, `setup/vllm-parsa`, `setup/vllm-beam`, and `setup/vllm-ascend`, and builds/runs `vllm-ascend-parsa`.
Both legacy PARSA and BEAM builds select `Dockerfile.0.11`. Its recipe is based on the last pre-dependency-policy recipe at `bcbfb01`, with application dependencies inlined; the historical recipe has not been build-verified against the current runtime.

### v0.23 PARSA profile — vendor-preserving build

The v0.23 build path targets `vllm-ascend-parsa:0.23.0` and uses the matching vendor image.

The base image is the matching vendor image, `quay.io/ascend/vllm-ascend:v0.23.0`.
The profile uses these exact release pins:

| Component | Selected value |
| --- | --- |
| Base image | `quay.io/ascend/vllm-ascend:v0.23.0` |
| vLLM | `v0.23.0`, commit `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` |
| vLLM-Ascend | `v0.23.0`, commit `5cb98caaadeff42b5b62b996e34bb2aaa29d20fd` |
| Application dependencies | Inlined in `Dockerfile.0.23` |
| PARSA mailbox | `setup/parsa-vllm-0.23.0.patch` |
| Intended image | `vllm-ascend-parsa:0.23.0` |

```bash
python3 setup/deploy.py prepare-vllm --version 0.23
(cd setup && python3 deploy.py build-image parsa --version 0.23)
python3 setup/deploy.py run parsa --version 0.23 --devices 0,1
```

Preparation checks the source revisions and applies the v0.23 mailbox in isolated `setup/vllm-0.23.0`, `setup/vllm-parsa-0.23.0`, and `setup/vllm-ascend-0.23.0` trees.
It requires unused upstream/PARSA paths and accepts an existing Ascend checkout only when it is clean at the pinned revision.
Select `--version 0.23` on every action; preparation does not change the default for later commands.
The v0.23 selection is PARSA-only; other backends are rejected before Docker. v0.11 remains the default; BEAM also selects the v0.11 recipe.

Use the profile's matching **CANN, PyTorch, torch-npu, Python, vLLM, and vLLM-Ascend versions** as one stack.
Do not mix v0.23 with the v0.11 base/built image or source directories, including through build overrides or runtime mounts.
The deployment CLI rejects v0.11 PARSA/Ascend source paths for v0.23 builds and mounts.
Source preparation does not certify source-code identity or a runnable image. Existing v0.23 images may still be selected with `run`, but hardware/runtime compatibility is unverified.

### v0.23 image dependencies

`Dockerfile.0.23` uses ordinary installer resolution, with NumPy `1.26.4`, Torch `2.10.0`, Transformers `5.5.4`, FastAPI `>=0.115,<0.124`, and PyArrow `24.0.0` in its application install. The matching vendor base supplies the Ascend runtime and hardware companions; the recipe does not add OpenCV or Triton to that install. It retains `--no-deps` and `--no-build-isolation` only for copied core/plugin editables, with SCM versioning for both. It does not bootstrap Python 3.11 or CANN.

The [upstream v0.27.1 amd64 manifest](https://quay.io/repository/ascend/vllm-ascend/manifest/sha256:34a9fafd839cf5f9da9528130c9e24e81bcf97b0100e7f20c8eb10cc271c1b99) is a reference only, not the v0.23 base or digest; the base remains `quay.io/ascend/vllm-ascend:v0.23.0`.

Removing dependency-policy checks does not repair inherited vendor issues, including the TE `cp311`/Python 3.12 mismatch or known package conflicts. Build success and runtime compatibility are unverified; deployment tests remain authored but unrun.

### Validation boundaries

When test execution is authorized, run the mocked deployment checks from the repository root:

```bash
python3 -m unittest tests.test_setup_deploy -v
git diff --check
```

Hardware validation is separate and requires a compatible Ascend A2 host with the matching stack above.
After building the v0.23 image, check serving help and start an enabled KV-projection server.
When forwarding vLLM flags after `--`, place deployment options before the `parsa` positional argument:

```bash
python3 setup/deploy.py run --version 0.23 --devices 0,1 parsa -- --help
python3 setup/deploy.py run --version 0.23 --devices 0,1 --port 8000 parsa -- \
  --enable-parsa-kv-projection --parsa-engine-id parsa-0.23
```

Then send a completion request and exercise `GET /admin/parsa/kv-projection`, `X-Energy-Aware-*` request metadata propagation, concurrent requests, cancellation, and endpoint error handling (404 when disabled, 503 when the snapshot is unavailable).
Record the exact package versions, base/built image digests, device IDs/model, driver/CANN version, and each outcome with the experiment evidence.
These runtime checks have not been performed here.

## Task commands

### Local serving profiles and benchmark conditions

Every experiment must apply an active model/running-vLLM profile from `src/serving_profiles.json`; there is no opt-out.
All six revision-2 entries are ready: Qwen3 4B/8B/14B × running vLLM 0.11.0/0.23.0.
Canonical `Qwen/Qwen3-*` names and exact local basenames such as `/models/qwen3-8b` are supported.
Unknown models, missing profiles, and missing/unsupported actual vLLM releases raise before output creation or launch; environment version labels are not authoritative.
Model and actual vLLM release select the profile; hardware, precision, configured TP, and vLLM-Ascend versions are provenance, with no exact BF16/TP2 or Ascend-version-pair gate.

| Running vLLM | Active defaults for all three sizes |
| --- | --- |
| 0.11.0 | `PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256` |
| 0.23.0 | `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`, `--gpu-memory-utilization 0.9` |

These are documented adaptations of the recorded [0.11 SingleNPU tutorial](https://docs.vllm.ai/projects/ascend/en/v0.11.0/tutorials/single_npu.html) and [0.23 Qwen3-Dense tutorial](https://docs.vllm.ai/projects/ascend/en/v0.23.0/tutorials/models/Qwen3-Dense.html), generalized beyond their original model/scenario examples, without claims of optimality.
The 0.11 validation version is unknown; the 0.23 tutorial reports validation on 0.21.0 and is reference evidence rather than runtime validation here.

All five experiment runners use the same mandatory planner:

- `experiments/benchmarks.py`: condition-qualified benchmark output directories and expected child condition.
- `experiments/kv_baseline.py`: the same profile planning and child-condition contract for KV runs.
- `experiments/specialization_gap.py`: all selected banks/repetitions planned before writes; frozen profile identity in `experiment.json`.
- `experiments/op_conflict.py`: frozen characterization selection plus mandatory run profile identity in `experiment.json`.
- `experiments/azure_mix.py`: direct interpreter-bound serving, runtime-origin preflight, and a `.serving.json` sidecar.

`run.py` is the shared child launcher for the first four runners.
Optional `VLLM_SERVING_PROFILE_QUALIFIERS='{"hardware":"Ascend-910B","precision":"bf16"}'` records context only; the JSON object cannot contain `tp`, which comes from authoritative runner configuration.
Hardware and precision are never inferred from model basenames.
Precedence is **profile defaults < ambient argv/environment < config serving intent**.
Use `VLLM_SERVE_EXTRA_ARGS` for ambient argv, `VLLM_SERVING_OVERRIDES` for ambient raw JSON intent, or an optional config field:

```json
{
  "serving": {
    "args": {"gpu-memory-utilization": 0.85},
    "env": {"PYTORCH_NPU_ALLOC_CONF": "max_split_size_mb:128"}
  }
}
```

The shared benchmark/KV sweep configs support **root < lane < workload** serving overrides; gap/op configs accept the same `serving.args` option map and `serving.env` string map at experiment scope.
`compilation-config` and `additional-config` JSON objects merge by leaf; individual tuning flags/environment values can override defaults.
Ambient `-O3`, `-O=3`, and `-O 3` share the canonical `--compilation-config` key, so experiment intent takes precedence without a leftover short flag.
Serving intent cannot override authoritative model, port, parallelism, batch/sequence limits, devices, scheduling/chunked-prefill/trust settings, or runner lifecycle fields, including ambiguous flag aliases/abbreviations.
Credentials remain inherited outside serving configuration and are not recorded.
Explicit URL userinfo (including schemeless authenticated proxies) is rejected without echoing credential values; inherited authenticated proxies and credential-bearing tuning values still reach subprocesses but are excluded from provenance and condition keys.
Parents transport normalized raw intent under `VLLM_SERVING_OVERRIDES`; resolved defaults are applied only to final server environments.

Benchmark directories keep their existing slug and append `-sp-<24 hex digits>`, retaining `result.json` and the `profile-request-shape-*` prefix.
The parent passes `VLLM_EXPECTED_SERVING_CONDITION` and `run.py` rejects mismatches before lifecycle effects.
Results store authoritative `serving_profile` provenance and `serving_condition` after experiment metadata; these two keys are reserved in `EXPERIMENT_METADATA` and benchmark config metadata.
Provenance includes profile/source identity, launcher-interpreter distribution versions and package-origin anchors, effective extra args, relevant tuning environment, qualifiers, and each locally launched lane's full command, working directory and device/tuning environment.
Before cleanup, frequency setting or service launch, `run.py` probes package metadata using `sys.executable -c` in the server's unchanged `VLLM_ROOT` working directory with the projected `PYTHONPATH` and final profile environment.
This standard-library-only discovery does not import vLLM, vLLM-Ascend, torch or torch-npu; both package origin/root paths and installed distribution versions must exactly match the parent anchors.
Missing or invalid discovery, a mismatch, a failed probe or its ten-second timeout aborts preflight; `VLLM_VERSION` is not proof of the runtime package context.
The verified environment is reused for launch, changing only each lane's Ascend visibility, and successful preflight records `verified_package_origins` and `runtime_verified` provenance.
Package origins also participate in the condition key in both the benchmark parent and runner child.
Mandatory profiles require ownership of every serving lane: `SKIP_VLLM`, any per-lane skip control, or a planned `start=False` lane fails before runtime preflight, state writes, or lifecycle actions.
External-server attestation is unsupported; `SKIP_ROUTER` remains available for the KV baseline's direct client connection.
The runner owns `--result-json`; configure its path through `RESULT_JSON`, not a client-extra flag or abbreviation.
It compares output device/inode/size/nanosecond modification and change times immediately around the client process, before timestamps or merges.
An unchanged or absent JSON receives no timestamps, power/KV merges, experiment metadata, or profile stamp; monitors and services are cleaned up, and a zero client exit without fresh output becomes an orchestration failure.
Fresh JSON must be an object (with an array of request objects when present); malformed fresh output is left intact for diagnosis.
Valid fresh failed-request results retain their client exit status and receive current provenance.

Gap/op capture one runtime anchor per run invocation and freeze resolved profile ID/hash, condition, effective extra argv, relevant environment, actual versions, and source/package-origin provenance in `experiment.json`, alongside the raw normalized config.
Every candidate carries that condition into its manifest and `VLLM_EXPECTED_SERVING_CONDITION`; frequency/max-sequence banks remain under one profile condition.
Changed identities and older output roots missing the new frozen identity require a **fresh output root** before persistence; historical manifests are not rewritten.
Aggregation rejects result conditions differing from the frozen experiment before CSV writes, preventing cross-condition repetition pooling while retaining the existing raw energy, cohort, search, and SLO methodology.

`plots.request_shape_characterization` rejects mixed serving conditions after valid-shape/positive-request filtering; select `--serving-condition sp-<24 hex digits>` or `--serving-condition legacy` for older results without metadata.
The latency loaders accept legacy and `-sp-<24 hex digits>` directories, carry `serving_condition` into rows, and reject directory/JSON condition mismatches.
Use `serving_condition="sp-<24 hex digits>"` or `serving_condition="legacy"` with `load_latency_experiments_dataframe` / `load_latency_json_results` when the input contains multiple conditions; unfiltered mixed input raises before downstream averaging.

For offline OP characterization selection, `experiments.op_conflict --select` / `--inspect` accepts `--serving-condition sp-<24 hex digits>` (or `legacy`).
Without a selector, exactly one condition must remain after model, TP, fixed batching and frequency relevance checks; an ambiguity lists the available conditions.
Filtering happens before grouping repetitions, so partial repetition sets from different conditions cannot jointly satisfy the configured five-repetition eligibility requirement.
New `selection.json` files freeze the chosen characterization condition; older files without the field remain legacy and retain their frozen selection hashes.
An explicit selector during inspection must match an existing frozen selection, and the option is rejected for run, aggregate and other modes.

Focused integration checks were written test-first but **not run**; validation here is static inspection and `git diff --check` only.
The package-context probe and hardware/runtime checks have **not been run** here.

### Run Azure-mix and record actual token lengths

Run inside the container from `/workspace`.
This starts vLLM on all `ASCEND_RT_VISIBLE_DEVICES` with matching tensor parallelism, reads the existing schedule without rebuilding it, writes `index,input_length,output_length` to `./datasets/azure_mix_results.csv`, then stops the vLLM process it started.
The server command uses the launcher's `sys.executable -m vllm.entrypoints.cli.main`; package-origin preflight runs in the repository root with the exact final server environment before directories/logs are created.
That environment is reused for the server; serving overrides are not passed to the client.
`./datasets/azure_mix_results.serving.json` records `serving_profile`, `serving_condition`, full `argv`, `cwd`, and credential-free device/tuning `effective_env`; it is provenance, not a measurement result.
The sidecar is written only after a fresh CSV, using a unique temporary sibling and atomic replacement, and binds its provenance to the CSV's exact bytes through `result_csv_sha256`, `result_csv_bytes`, and `client_returncode`.
Fresh failed-request CSVs receive that binding before a nonzero client exit is propagated; readiness failure or absent/unchanged output leaves the previous CSV/sidecar pair untouched, and zero exit without fresh CSV fails orchestration.
No production sidecar reader currently makes profile claims; `experiments.azure_mix.verified_serving_sidecar(result_csv)` returns the bound record or `None` for missing, legacy unbound, malformed, or content-mismatched provenance.
Readers must verify the content binding before associating a CSV with a serving profile; a matching binding establishes provenance for those bytes, not success of a later invocation.
Use `--serving-overrides` to layer JSON args/env intent above ambient intent:

```bash
python3 experiments/azure_mix.py \
  --schedule /datasets/azure-mix/scheduled_prompts.csv \
  --result-csv ./datasets/azure_mix_results.csv \
  --serving-overrides '{"args":{"gpu-memory-utilization":0.85},"env":{"PYTORCH_NPU_ALLOC_CONF":"max_split_size_mb:128"}}'
```

### Scheduled prompt and actual generated-token histogram

Run inside the container from `/workspace`.
This joins the schedule to actual token lengths and writes per-workload histograms with CDFs.

```bash
python3 -m plots.plot_token_histogram \
  --schedule /datasets/azure-mix/scheduled_prompts.csv \
  --result-csv ./datasets/azure_mix_results.csv \
  --output /workspace/plots/output/token_histogram.png
```
