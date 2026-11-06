# BEAM vs VLLM Evaluation

Your task is to run an experiments as per the following instructions.
Pick different workloads for each experimental run.
I would like to see results of different experimental runs / workloads.

## Setup (Example Commands)

Generate BEAM profile:
```bash
python3 tests/beam_wide_profile.py \
    --output setup/beam_profiles/910b3-qwen3-4b-wide-toy.csv
```

Command to start BEAM:
```bash
cd setup
python3 deploy.py run \
    --port 8000 \
    --version 0.23 \
    --devices 0 \
    --model Qwen__Qwen3-4B \
    --vllm ./vllm-beam-npu-0.23.0 \
    beam -- \
    --use-s1 \
    --use-s2 \
    --beam-gpu-name 910B3 \
    --beam-profile-path /workspace/setup/beam_profiles/910b3-qwen3-4b-wide-toy.csv \
    --beam-npu-device-map '{"0":0}' \
    --beam-ttft-slo 0.6 \
    --beam-tbt-slo 0.12 \
    --max-num-batched-tokens 2048
```

Run Test:
```bash
python3 tests/test_beam_dvfs.py \
    --url http://127.0.0.1:8000 \
    --device 0 \
    --mode phases \
    --phase-seconds 25 \
    --prefill-rps 1 \
    --decode-rps 0.5 \
    --sample-interval 0.25 \
    --out /tmp/beam-dvfs-wide
```

## Experimental Methodology

Compare both BEAM-vLLM-0.23 and vLLM-0.23 containers.

Run with Tensor Parallelism = 2 (Use NPUs 0 and 1 for the experiments.)

Model to be used: Qwen3-8B (check /data1/models/Qwen__Qwen3-8B/).

Select 2 operating points: (<comcurrency-cap>,<DVFS-frequency-MHz>)
Prefill-Heavy Operating Point: (10,1700)
Decode-Heavy Operating Point: (200, 1300)

Metrics To Monitor: e2e NPU energy, TTFT, TPOT, TTFT, DVFS frequency of each NPU, etc.
Collect these metrics every 0.5 second

Worlkoads: Look at the workloads inside in experiments directory.
Try benchmarks.py with benchmarks.json.
Also see if you can use the azure_mix.py script (not sure if it needs a json file as arg)

## Deliverables

Create one folder inside .agents/beam-vs-vllm/ for each workload-run.
Inside it place all raw data
- metrics dumped to csv files
- Create an analysis.md which analyses the data, explains the workload and compares vLLM and BEAM for that.
- Generate matplotlib plots for various metrics.