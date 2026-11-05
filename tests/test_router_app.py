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

import unittest
from unittest.mock import AsyncMock, patch

from src.controller import EnergyAwareController, SLO
from src.presets import Preset
from src.profile_table import DictProfileTable
from src.router_app import EnergyAwareRouter, create_app


def _make_controller():
    profile = DictProfileTable.from_mapping(
        {"lane_a": {"a_ttft": 0.10, "a_tbt": 0.01, "a_ep": 2.0, "a_ed": 0.05}}
    )
    return EnergyAwareController(
        presets=[Preset("lane_a", "http://127.0.0.1:9999", 4096, max_num_seqs=2)],
        profile=profile,
        slo=SLO(20.0),
        routing_policy="random",
    )


class RouterAppTests(unittest.TestCase):
    def test_chat_completions_does_not_require_query_request_param(self):
        try:
            from fastapi.testclient import TestClient
            from starlette.responses import JSONResponse
        except ImportError:
            self.skipTest("fastapi not installed")

        controller = _make_controller()
        app = create_app(controller)

        async def fake_handle(_request, _path):
            return JSONResponse({"ok": True})

        with patch.object(EnergyAwareRouter, "handle_completion", new=AsyncMock(side_effect=fake_handle)):
            with TestClient(app) as client:
                response = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "demo",
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": False,
                    },
                )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {"ok": True})


if __name__ == "__main__":
    unittest.main()
