#!/usr/bin/env python3
"""Idempotently add/remove one diagnostic hook to BEAM's Ascend worker.

Run on HOST before launching:
    python3 tests/beam_enable_logging.py --tree setup/vllm-beam-npu-0.23.0
    python3 tests/beam_enable_logging.py --tree setup/vllm-beam-npu-0.23.0 --undo

Do NOT run `git checkout -- <worker>` to undo: your prepared vLLM checkout
may carry the BEAM patch as uncommitted changes on top of vanilla vLLM.
"""

import argparse
from pathlib import Path

REL = Path("vllm/v1/worker/beam_ascend_worker.py")
ORIGINAL = '''            observed = self._get_beam_clock_controller().set_clock(requested)
            if self.beam_profiler is not None:
'''
INSTRUMENTED = '''            logger.warning(
                "[BEAM-DVFS] REQUEST rank=%s step=%s phase=%s tokens=%s requested_mhz=%s",
                self.rank, getattr(scheduler_output, "beam_step", None),
                getattr(scheduler_output, "beam_phase", None), total, requested,
            )
            observed = self._get_beam_clock_controller().set_clock(requested)
            logger.warning(
                "[BEAM-DVFS] RESULT rank=%s requested_mhz=%s observed_mhz=%s matched=%s",
                self.rank, requested, observed, observed == requested,
            )
            if self.beam_profiler is not None:
'''


def edit(tree: Path, undo: bool):
    target = tree / REL
    if not target.is_file():
        raise SystemExit(f"Worker file not found: {target}")
    source = target.read_text(encoding="utf-8")
    before, after = (INSTRUMENTED, ORIGINAL) if undo else (ORIGINAL, INSTRUMENTED)
    if before not in source:
        if after in source:
            print("Already " + ("unpatched" if undo else "instrumented") + f": {target}")
            return
        raise SystemExit("Expected BEAM worker code not found; refusing to edit unknown version")
    if source.count(before) != 1:
        raise SystemExit(f"Expected exactly one match; found {source.count(before)}")
    target.write_text(source.replace(before, after, 1), encoding="utf-8")
    print(("Removed" if undo else "Installed") + f" BEAM DVFS logging in {target}")
    print("This changes Python source only. Pass --vllm <tree> to deploy.py run, or rebuild image.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tree", type=Path, required=True, help="Prepared vllm-beam-npu-0.23.0 source directory")
    p.add_argument("--undo", action="store_true", help="Remove just this logging hook")
    args = p.parse_args()
    edit(args.tree, args.undo)


if __name__ == "__main__":
    main()
