BEAM NPU toy DVFS smoke test (Qwen3-4B / Ascend 910B3 / TP1 PP1)
===============================================================
These files are deliberately SYNTHETIC. They test policy selection and
clock-control plumbing, not speed, energy efficiency, or SLO compliance.

Extract zip into the parsa repository root (not setup/):
  unzip -o beam_dvfs_smoke_bundle.zip -d /data2/alan/parsa
  cd /data2/alan/parsa

Optional regenerate profile, or change only catalog clock pair:
  python3 tests/beam_toy_profile.py --output setup/beam_profiles/910b3-qwen3-4b-toy.csv
  # python3 tests/beam_toy_profile.py --output ... --low-clock 900 --high-clock 1650

Pure-CPU mock of DSMI controller, no hardware writes:
  python3 -m unittest tests.test_beam_npu_clock_mock -v

Instrument the prepared BEAM vLLM worker (Python-only edit):
  python3 tests/beam_enable_logging.py --tree setup/vllm-beam-npu-0.23.0

Start server with the edited source mounted (stop any old server first):
  cd setup
  python3 deploy.py run --port 8000 --version 0.23 --devices 0 \
    --model Qwen__Qwen3-4B --vllm ./vllm-beam-npu-0.23.0 beam -- \
    --use-s1 --use-s2 --beam-gpu-name 910B3 \
    --beam-profile-path /workspace/setup/beam_profiles/910b3-qwen3-4b-toy.csv \
    --beam-npu-device-map '{"0":0}' \
    --beam-ttft-slo 0.6 --beam-tbt-slo 0.12 \
    --max-num-batched-tokens 2048

With the container up, confirm that the ACTUAL patched BEAM policy makes
our expected choices WITHOUT changing any NPU clock:
  CID=$(docker ps -q --filter ancestor=vllm-ascend-beam:0.23.0 | head -n1)
  docker exec "$CID" python /workspace/tests/beam_toy_profile.py \
    --check-only \
    --output /workspace/setup/beam_profiles/910b3-qwen3-4b-toy.csv
  # Expected: 64-token prefill -> 1500; 2048-token prefill -> 1650;
  #           decode -> 1500 MHz.

In another host terminal run the existing user test:
  cd /data2/alan/parsa
  python3 tests/test_beam_dvfs.py --url http://127.0.0.1:8000 \
    --device 0 --mode phases --phase-seconds 25 \
    --prefill-rps 1 --decode-rps 0.5 --sample-interval 0.25 \
    --out /tmp/beam-dvfs-toy

Inspect BEAM worker logs:
  docker logs "$CID" 2>&1 | grep 'BEAM-DVFS'
Expected if active: REQUEST and RESULT rows with clock requests 1500/1650.
If RESULT observed=1800 despite requested=1500, firmware/DSMI has not honored
this clock request. The code does not currently raise on such a mismatch.

Remove only the optional instrumentation with:
  python3 tests/beam_enable_logging.py --tree setup/vllm-beam-npu-0.23.0 --undo
DO NOT use `git checkout -- workerfile` to undo if BEAM patches were applied as
uncommitted raw diffs on a vanilla vLLM base.

Safety: Use an exclusively assigned NPU, watch health and thermal limits,
and check the idle frequency after clean shutdown. The current NPU controller
restores RATED MHz (1800 on this host) rather than proven auto-DVFS mode.
