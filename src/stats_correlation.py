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

import math


def _mean(values):
    values = list(values or [])
    if not values:
        return None
    return sum(values) / float(len(values))


def pearson(x_values, y_values):
    x_values = [float(item) for item in list(x_values or [])]
    y_values = [float(item) for item in list(y_values or [])]
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return None
    mean_x = _mean(x_values)
    mean_y = _mean(y_values)
    cov = 0.0
    var_x = 0.0
    var_y = 0.0
    for x_item, y_item in zip(x_values, y_values):
        dx = x_item - mean_x
        dy = y_item - mean_y
        cov += dx * dy
        var_x += dx * dx
        var_y += dy * dy
    if var_x <= 0 or var_y <= 0:
        return None
    return cov / math.sqrt(var_x * var_y)


def _rank(values):
    values = list(values or [])
    indexed = sorted((float(value), index) for index, value in enumerate(values))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][0] == indexed[cursor][0]:
            end += 1
        rank_value = (cursor + end - 1) / 2.0 + 1.0
        for _, original_index in indexed[cursor:end]:
            ranks[original_index] = rank_value
        cursor = end
    return ranks


def spearman(x_values, y_values):
    x_values = list(x_values or [])
    y_values = list(y_values or [])
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return None
    return pearson(_rank(x_values), _rank(y_values))


def summarize_pairs(x_values, y_values):
    return {
        "pearson": pearson(x_values, y_values),
        "spearman": spearman(x_values, y_values),
        "count": len(list(x_values or [])),
    }
