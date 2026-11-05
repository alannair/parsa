# Copyright (C) 2026 M.R. Siavash Katebzadeh <mr@katebzadeh.xyz>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.


import argparse
import os
import signal
import sys
import time
from pathlib import Path


LEGACY_PID_FILES = (
    "/tmp/energy_aware_demo_vllm_p.pid",
    "/tmp/energy_aware_demo_vllm_d.pid",
    "/tmp/energy_aware_demo_router.pid",
)


def _read_pid(pid_file):
    try:
        text = pid_file.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _pid_alive(pid):
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def kill_pid(pid, force):
    if pid <= 0:
        return
    if not _pid_alive(pid):
        return
    sig = signal.SIGKILL if force else signal.SIGTERM
    try:
        os.kill(pid, sig)
    except OSError:
        pass


def _collect_pids_from_files(pid_files):
    pids = []
    seen = set()
    for pid_file in pid_files:
        pid = _read_pid(pid_file)
        if pid is not None and pid not in seen:
            seen.add(pid)
            pids.append(pid)
    return pids


def _unlink_pid_files(pid_files):
    for pid_file in pid_files:
        try:
            pid_file.unlink()
        except OSError:
            pass


def _wait_for_pids(pids, timeout_s, poll_s=0.5):
    if timeout_s <= 0 or not pids:
        return [pid for pid in pids if _pid_alive(pid)]
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        alive = [pid for pid in pids if _pid_alive(pid)]
        if not alive:
            return []
        time.sleep(poll_s)
    return [pid for pid in pids if _pid_alive(pid)]


def kill_pid_file(pid_file, force):
    if not pid_file.is_file():
        return None
    pid = _read_pid(pid_file)
    if pid is not None:
        kill_pid(pid, force)
    try:
        pid_file.unlink()
    except OSError:
        pass
    return pid


def collect_pid_files(demo_state_dir):
    pid_files = []
    seen = set()

    def add(path):
        key = str(path)
        if key in seen:
            return
        seen.add(key)
        pid_files.append(path)

    list_path = demo_state_dir / "pid_files.list"
    if list_path.is_file():
        for line in list_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                add(Path(line))

    for path in sorted(demo_state_dir.glob("*.pid")):
        add(path)

    for legacy in LEGACY_PID_FILES:
        add(Path(legacy))

    return pid_files


def kill_all(demo_state_dir=None, force=False, wait_timeout_s=60.0):
    state_dir = Path(
        os.path.abspath(os.path.expanduser(str(demo_state_dir or os.environ.get("DEMO_STATE_DIR", "/tmp/energy_aware_demo"))))
    )
    pid_files = collect_pid_files(state_dir)
    pids = _collect_pids_from_files(pid_files)
    _unlink_pid_files(pid_files)

    if pids:
        if force:
            for pid in pids:
                kill_pid(pid, True)
        else:
            for pid in pids:
                kill_pid(pid, False)
            alive = _wait_for_pids(pids, wait_timeout_s)
            for pid in alive:
                kill_pid(pid, True)
            _wait_for_pids(alive, min(10.0, wait_timeout_s))

    for stale in (state_dir / "all.pids", state_dir / "pid_files.list", Path("/tmp/energy_aware_demo.pids")):
        try:
            stale.unlink()
        except OSError:
            pass


def build_parser():
    parser = argparse.ArgumentParser(prog="kill.py")
    parser.add_argument(
        "--demo-state-dir",
        default=os.environ.get("DEMO_STATE_DIR", "/tmp/energy_aware_demo"),
        help="State directory written by run.py (default: $DEMO_STATE_DIR or /tmp/energy_aware_demo)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Send SIGKILL instead of SIGTERM",
    )
    parser.add_argument(
        "--wait-timeout-s",
        type=float,
        default=60.0,
        help="Seconds to wait for graceful shutdown before SIGKILL (default: 60)",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    kill_all(args.demo_state_dir, force=bool(args.force), wait_timeout_s=float(args.wait_timeout_s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
