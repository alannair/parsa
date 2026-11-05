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

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import kill as kill_module


class KillTests(unittest.TestCase):
    def test_collect_pid_files_includes_all_pid_glob(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            (state_dir / "vllm_lane_d.pid").write_text("101\n", encoding="utf-8")
            (state_dir / "router.pid").write_text("202\n", encoding="utf-8")
            (state_dir / "power_monitor.pid").write_text("303\n", encoding="utf-8")
            pid_files = kill_module.collect_pid_files(state_dir)
            names = {path.name for path in pid_files}
            self.assertIn("vllm_lane_d.pid", names)
            self.assertIn("router.pid", names)
            self.assertIn("power_monitor.pid", names)

    def test_kill_all_waits_then_escalates_to_sigkill(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            (state_dir / "router.pid").write_text("4242\n", encoding="utf-8")
            signals = []

            def fake_kill(pid, sig):
                signals.append((pid, sig))

            alive = {4242}

            def fake_alive(pid):
                return pid in alive

            with patch.object(kill_module.os, "kill", side_effect=fake_kill):
                with patch.object(kill_module, "_pid_alive", side_effect=fake_alive):
                    with patch.object(kill_module.time, "sleep", return_value=None):
                        kill_module.kill_all(state_dir, wait_timeout_s=1.0)

            self.assertIn((4242, kill_module.signal.SIGTERM), signals)
            self.assertIn((4242, kill_module.signal.SIGKILL), signals)
            self.assertFalse((state_dir / "router.pid").exists())

    def test_kill_all_clears_pid_tracking_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            (state_dir / "pid_files.list").write_text("/tmp/example.pid\n", encoding="utf-8")
            (state_dir / "all.pids").write_text("1\n", encoding="utf-8")
            with patch.object(kill_module, "_pid_alive", return_value=False):
                kill_module.kill_all(state_dir, wait_timeout_s=0.0)
            self.assertFalse((state_dir / "pid_files.list").exists())
            self.assertFalse((state_dir / "all.pids").exists())


if __name__ == "__main__":
    unittest.main()
