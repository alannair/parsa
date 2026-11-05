# BEAM v0.23: two debug layers, one Ascend preparation

The two patches are **ordered, consolidated Git diffs**, not `git am` mailboxes:

| Order | Asset | Input | Result |
|---|---|---|---|
| 1 | `beam-vllm-0.23.0.patch` | Official vanilla vLLM `v0.23.0` | BEAM ported to the v0.23 interfaces |
| 2 | `beam-npu-vllm-0.23.0.patch` | Exactly the result of patch 1 | BEAM with the Ascend/NPU adaptation |

Patch 2 does **not** apply directly to vanilla vLLM. The split exists for
debugging; normal preparation applies both to the **same** BEAM worktree. It
does not create or build an intermediate GPU-only BEAM version.

Existing PARSA preparation and the legacy v0.11 patch files remain unchanged.
In particular, do not substitute the old `beam-vllm.patch` for either new asset.

## Exact boundaries

- Vanilla vLLM tag: `v0.23.0`.
- Vanilla commit: `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`.
- Patch 1 source snapshot: `38d461ca6a4694d944b8496b36091ff6c42ef6ad`.
- Patch 1 result Git tree: `8e32990cddc8af35c87ef941a42c833a8c141e34`.
- Patch 2 result Git tree: `e68dda0c15db64691dbae39c81c4bbdf80dd35cb`.
- Required vLLM-Ascend tag: `v0.23.0`.
- Required vLLM-Ascend commit: `5cb98caaadeff42b5b62b996e34bb2aaa29d20fd`.

The original artifact's closest demonstrated upstream source match was v0.11.2;
the patch names refer to the **target** version, not a claim that the original
artifact was exactly the v0.11.0 tag.

Patch 1 contains 71 paths. Patch 2 contains 20 paths: 16 updates and four new
files, including all three Ascend worker modules. Full object IDs and file modes
are retained. The overlay was exported through a temporary Git index; the source
checkout's real index, branch and working files were not changed by export.
Unrelated `.serena/` state, local results and credentials are excluded.

Integrity references are Git **blob** IDs of the patch assets, not SHA-256 sums
or evidence of successful application:

| Asset | Git blob ID |
|---|---|
| `beam-vllm-0.23.0.patch` | `11d72426c9f2cbef302359506592fc1299b014d2` |
| `beam-npu-vllm-0.23.0.patch` | `a9168dbb4e6f93f60c8ccf7a3bd5997226388d20` |

## Preparation — later-only commands

Run from this `setup` directory, or use `make -C setup` from `parsa/code`:

```bash
python deploy.py prepare-vllm --version 0.23
# Equivalent:
make beam-prepare
```

The preparation flow retains the existing PARSA worktree and additionally
creates just one BEAM worktree:

```text
vllm-0.23.0/                 pinned upstream repository
vllm-parsa-0.23.0/           existing PARSA preparation
vllm-beam-npu-0.23.0/        patch 1, then patch 2; branch beam-port-v0.23
vllm-ascend-0.23.0/          shared clean, pinned Ascend backend
```

All required patch files are checked before cloning. Existing destination paths
are refused instead of overwritten. The upstream and Ascend revisions are
checked against the immutable pins. Only the core vLLM worktree receives these
two patches; the pinned Ascend backend is not modified.

`apply_vllm_patch` accepts the new raw Git diffs as one layer each. Legacy
format-patch mailboxes still use the existing per-message flow. Future setup
runs use Git application checks; no such checks or preparation were run while
creating these assets.

### Manual layer debugging

In a separate fresh checkout at the exact vanilla commit, use the branch name
`beam-port-v0.23` and apply the files **separately and in order**:

```bash
git -C /absolute/fresh-vllm switch -c beam-port-v0.23 v0.23.0
git -C /absolute/fresh-vllm apply /absolute/setup/beam-vllm-0.23.0.patch
# Inspect the intermediate source here; no GPU-only build is required.
git -C /absolute/fresh-vllm apply /absolute/setup/beam-npu-vllm-0.23.0.patch
```

These commands are instructions for later use, not executed validation results.
Applying raw diffs does not reproduce the historical commits: HEAD can remain
at the upstream commit while the patch contents are working-tree changes.

If layer 2 fails, preparation stops and leaves layer 1 available for inspection.
Do not blindly replay layer 1 after layer 2: the overlay intentionally changes
its postimages. The setup is fresh-only, not a general resume mechanism. Save
and review partial/local changes before explicitly choosing a new preparation
location or cleaning up worktrees.

## Build and run — final NPU image only

```bash
python deploy.py build-image beam --version 0.23 --wheelhouse ./wheelhouse
# Equivalent with the default ./wheelhouse location:
make beam-image-0.23
```

The build selects `vllm-beam-npu-0.23.0`, the shared pinned Ascend checkout and
`Dockerfile.0.23`. The final image is `vllm-ascend-beam:0.23.0`. It reuses the
existing v0.23 offline PyArrow repair: provide the PyArrow 24.0.0 `aarch64`
manylinux wheel matching the base image's CPython version. The setup script
checks the image's Python ABI before accepting the wheel. A `-pre-pyarrow` tag
is a dependency-repair stage of the **NPU** image, not a GPU-only BEAM build.

`--offline-from` is accepted for BEAM only when the existing local image's
version, upstream/core/Ascend revisions and exact ordered patch-list labels
match the BEAM profile. A vanilla or PARSA image is rejected before retagging.
These labels are provenance checks, not proof that the image runs correctly.

For example, after supplying a real model and matching, reviewed NPU profile:

```bash
python deploy.py run --version 0.23 --devices 0 --model qwen3-4b beam -- \
  --use-s1 --use-s2 \
  --beam-gpu-name 910B4 \
  --beam-profile-path /mnt/profiles/910B4-qwen3-tp1-pp1.csv
```

Deployment options precede `beam`; native vLLM arguments follow `beam --`.
The example profile path is an operator-supplied placeholder in the existing
`/mnt` mount, not bundled/calibrated data. The setup does not silently enable
S1/S2 or invent a profile when no BEAM mode arguments are supplied.

BEAM v0.23 NPU launch sets `VLLM_USE_V2_MODEL_RUNNER=0` and supplies
`--enforce-eager --no-async-scheduling`. Conflicting explicit options are
rejected, not silently overridden. Numeric `ASCEND_RT_VISIBLE_DEVICES` metadata
is passed to the container; an optional `--beam-npu-device-map` can specify
physical DSMI IDs explicitly. The legacy `--beam-gpu-name` spelling is retained
for device-profile identity.

The bounded NPU candidate is synchronous V1/eager, native single-host uni/mp,
DP1 and no context/expert parallelism. `uni` requires TP×PP=1. Async/V2, ACL
graphs, alternate workers/schedulers, MoE/hybrid/SSM, speculation/LoRA and NPU
disaggregation/connectors are rejected. See the applied `BEAM_ASCEND_PORT.md`
for clock restoration, sampled-power semantics and hardware limitations.

Docker's existing build context excludes `.git`; copied serving sources do not
retain worktree Git metadata. The image supplies installation-version metadata,
but the optional experiment harness's stricter Git-source gates additionally
need accessible source repositories. Preparing/building the serving image is
not a claim that every provenance-checking experiment can run inside a
source-stripped container without that metadata.

## Verification boundary

The assets were exported with native Git diff/index operations and reviewed as
text, including their bases, full-index preimage/postimage chains and file
inventories. No patch was applied or application-checked here. No setup script,
test/collection, Python import, package installation, Docker build, inference,
benchmark, linter, formatter or GPU/NPU operation was executed. No commits were
created, and unrelated local files were left untouched.
