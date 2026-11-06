# Prefill Heavy Cap10

## Workload and setup

- Model: Qwen3-8B, vLLM 0.23.0, tensor parallelism 2 on NPU 0 and 1.
- Workload shape follows `experiments/req-shape-1.json`; synthetic prompts use the configured exact token counts.
- Synthetic request shape: 1000 input tokens, 1 output tokens; 200 offered requests per repetition.
- Server and client in-flight cap: 10.
- Vanilla vLLM clock policy: host default dynamic DVFS without a static clock lock; rated clock 1800 MHz, expected 800 MHz idle and up to 1800 MHz under load.
- BEAM configured TTFT/TBT SLOs: 0.600/0.120 s.
- BEAM S1/S2 used `tests/beam_wide_profile.py`'s synthetic profile; this was not calibrated for Qwen3-8B or TP2.
- Read-only DSMI frequency and whole-device power samples were collected every 0.5 s for each NPU.
- Sampled energy is trapezoidal integration of whole-device power; it includes idle/system power and is not an NPU energy-counter reading.
- Both NPUs were explicitly reset to rated 1800 MHz before and after each backend session; each reset verified 800 MHz at idle. Vanilla vLLM ran without a static clock lock.
- Every vanilla repetition sampled both 800 MHz idle and 1800 MHz loaded on NPUs 0 and 1.
- Warm-up traffic is retained under each backend's `warmup/` folder and excluded from the three measured repetitions.
- All BEAM requests completed, but its container required a forced stop (exit 137); the subsequent explicit clock reset succeeded and verified both devices at 800 MHz idle (`beam/clock_reset.log`).

The existing `experiments/benchmarks.json` does not expand: its lane sweeps omit the required `max_num_batched_tokens`. `experiments/azure_mix.py` was not run because `/data1/datasets/azure-mix/scheduled_prompts.csv` is absent.

## Results

| Backend | Rep | Completed/200 | Shape valid | Mean TTFT (s) | P95 TTFT (s) | Mean TPOT (s) | P95 TPOT (s) | Output tok/s | NPU energy (J) | Mean MHz NPU0/1 |
|---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| beam | rep1 | 200/200 | True | 0.427 | 0.707 | n/a | n/a | 22.134 | 3437.5 | 1353/1353 |
| beam | rep2 | 200/200 | True | 0.431 | 0.630 | n/a | n/a | 22.795 | 3436.8 | 1357/1357 |
| beam | rep3 | 200/200 | True | 0.437 | 0.610 | n/a | n/a | 22.398 | 3422.6 | 1357/1357 |
| vllm | rep1 | 200/200 | True | 0.340 | 0.475 | n/a | n/a | 28.600 | 3281.7 | 1371/1371 |
| vllm | rep2 | 200/200 | True | 0.329 | 0.455 | n/a | n/a | 29.518 | 3119.8 | 1415/1415 |
| vllm | rep3 | 200/200 | True | 0.330 | 0.512 | n/a | n/a | 29.511 | 3883.8 | 1257/1257 |

## Interpretation and limitations

- Compare measured TTFT/TPOT, throughput, sampled clocks, and integrated device power; BEAM's profile-estimated energy is not a measured value.
- The requested BEAM prefill point of 1700 MHz is outside the checked-in BEAM NPU clock catalog (900–1650 MHz). Vanilla was intentionally run without a static clock lock at the host-default dynamic setting, rated at 1800 MHz.
- Confirm the expected 800 MHz idle and 1800 MHz loaded clocks from the raw per-device telemetry; idle and boundary samples can lower the reported mean.
- TPOT is the request client's mean inter-content-chunk gap. It is not a hardware token-event counter.
- Repetitions are reported separately; first-run effects and run-to-run variability are retained rather than discarded.
- Inspect the saved request JSON/CSV and NPU sample CSVs for errors, token-shape validity, sampling gaps, and variability before drawing conclusions.

## Plots

![Request latency empirical CDF](latency_cdf.png)

![NPU frequency and power](frequency_power.png)

![Energy and throughput](energy_throughput.png)
