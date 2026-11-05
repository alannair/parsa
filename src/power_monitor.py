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

import time


class PowerSnapshot:

    __slots__ = ("timestamp_s", "cluster_power_w", "budget_w", "lane_power_w")

    def __init__(self, cluster_power_w=0.0, budget_w=None, lane_power_w=None, timestamp_s=None):
        self.timestamp_s = time.time() if timestamp_s is None else float(timestamp_s)
        self.cluster_power_w = float(cluster_power_w or 0.0)
        self.budget_w = None if budget_w is None else float(budget_w)
        self.lane_power_w = dict(lane_power_w or {})

    def lane_power(self, preset_id, default=0.0):
        return float(self.lane_power_w.get(preset_id, default) or 0.0)


class PowerMonitor:

    def snapshot(self):
        return PowerSnapshot()


class NullPowerMonitor(PowerMonitor):

    __slots__ = ("budget_w",)

    def __init__(self, budget_w=None):
        self.budget_w = budget_w

    def snapshot(self):
        return PowerSnapshot(cluster_power_w=0.0, budget_w=self.budget_w, lane_power_w={})


class StaticPowerMonitor(PowerMonitor):

    __slots__ = ("_snapshot",)

    def __init__(self, cluster_power_w=0.0, budget_w=None, lane_power_w=None):
        self._snapshot = PowerSnapshot(cluster_power_w=cluster_power_w, budget_w=budget_w, lane_power_w=lane_power_w)

    def set_snapshot(self, cluster_power_w=None, budget_w=None, lane_power_w=None):
        current = self._snapshot
        self._snapshot = PowerSnapshot(
            cluster_power_w=current.cluster_power_w if cluster_power_w is None else cluster_power_w,
            budget_w=current.budget_w if budget_w is None else budget_w,
            lane_power_w=current.lane_power_w if lane_power_w is None else lane_power_w,
        )

    def snapshot(self):
        current = self._snapshot
        return PowerSnapshot(
            cluster_power_w=current.cluster_power_w,
            budget_w=current.budget_w,
            lane_power_w=current.lane_power_w,
            timestamp_s=current.timestamp_s,
        )
