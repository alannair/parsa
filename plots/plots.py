import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
from matplotlib.lines import Line2D
from scipy import interpolate
from sklearn.inspection import PartialDependenceDisplay
from plots.style import (
    set_size,
    global_legend,
    style_legend,
    tick_style,
    cmap as style_cmap,
    plot_color,
    tokens_per_energy_heatmap_cmap,
    CONCURRENT_REQUESTS_LABEL,
)
from predictors.common import FEATURE_COLS, prediction_association_metrics, y_true_y_pred_from_train_result
feature_cols = FEATURE_COLS


def _legend(ax, *args, **kwargs):
    legend = ax.legend(*args, **kwargs)
    legend = style_legend(legend)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_alpha(1.0)
    return legend


def _figsize(width, height=None, fraction=1):
    return set_size(width, fraction=fraction, height=height)

def plot_epp_fixed_bs(df, bs=200, height=None):
    olens = [10, 100, 1000]
    ilens = [1, 10, 100, 1000]
    models = df['model'].unique()
    ncols = len(models)
    nrows = 1
    fig, axes = plt.subplots(nrows, ncols, figsize=_figsize(350, height=height))
    if ncols == 1:
        axes = [axes]
    for col, model in enumerate(models):
        axes[col].set_title(model)
        for i, olen in enumerate(olens):
            rows = []
            for ilen in ilens:
                filtered_df = df[(df['batch_size'] == str(bs)) & (df['model'] == model) & (df['input_size'] == str(ilen)) & (df['output_size'] == str(olen))]
                if not filtered_df.empty:
                    energy_joules = filtered_df['energy_joules'].iloc[0]
                    rows.append({'input_size': ilen, 'Y': energy_joules / 200})
            df_plot = pd.DataFrame(rows)
            if not df_plot.empty:
                df_plot = df_plot.sort_values('input_size')
                sns.lineplot(ax=axes[col], data=df_plot, x='input_size', y='Y', label=f'Output Tokens: {olen}', marker='o', color=plot_color(i))
    master_ax = axes[0]
    for ax in axes[1:]:
        ax.sharey(master_ax)
        ax.set_ylabel('')
        ax.set_xlabel('Input Length')
    master_ax.set_yscale('log')
    master_ax.set_xlabel('Input Tokens')
    master_ax.set_ylabel('Energy Per Prompt')
    fig.suptitle(f'{CONCURRENT_REQUESTS_LABEL} = {bs}')
    global_legend(axes[0], title='Output length')
    plt.tight_layout()
    plt.show()

def plot_epp_fixed_ilen_freq(df, ilen=10, bs=200, height=None):
    olens = [10, 100, 1000]
    freqs = [800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800]
    sns.set_theme(style='whitegrid')
    models = df['model'].unique()
    ncols = len(models)
    nrows = 1
    fig, axes = plt.subplots(nrows, ncols, figsize=_figsize(350, height=height))
    if ncols == 1:
        axes = [axes]
    for col, model in enumerate(models):
        axes[col].set_title(model)
        for i, olen in enumerate(olens):
            rows = []
            for freq in freqs:
                filtered_df = df[(df['batch_size'] == str(bs)) & (df['model'] == model) & (df['input_size'] == str(ilen)) & (df['output_size'] == str(olen)) & (df['frequency'] == str(freq))]
                if not filtered_df.empty:
                    energy_joules = filtered_df['energy_joules'].iloc[0]
                    rows.append({'freq': freq, 'Y': energy_joules / 200})
            df_plot = pd.DataFrame(rows)
            if not df_plot.empty:
                df_plot = df_plot.sort_values('freq')
                sns.lineplot(ax=axes[col], data=df_plot, x='freq', y='Y', label=f'Output Tokens: {olen}', marker='o', color=plot_color(i))
    master_ax = axes[0]
    for ax in axes[1:]:
        ax.sharey(master_ax)
        ax.set_ylabel('')
        ax.set_xlabel('Frequency')
    master_ax.set_xlabel('Frequency')
    master_ax.set_ylabel('Energy Per Prompt')
    fig.suptitle(f'{CONCURRENT_REQUESTS_LABEL} = {bs}')
    global_legend(axes[0], title='Output length')
    plt.tight_layout()
    plt.show()

def plot_frequency_throughput_energy(df, freq_col='frequency', throughput_col='tokens_per_second', energy_col='energy_joules', batch_col='batch_size', hue=None, title=None, freq_label='Frequency (MHz)', throughput_label='Throughput (tokens/s)', energy_per_prompt_label='Energy per prompt (J)', show=True, height=None):
    required = {freq_col, throughput_col, energy_col, batch_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f'Missing columns: {missing}')
    if hue is not None and hue not in df.columns:
        raise KeyError(f'hue column not in DataFrame: {hue!r}')
    d = df.copy()
    for c in required:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d.dropna(subset=list(required))
    d = d[d[batch_col] > 0]
    d = d.assign(energy_per_prompt=d[energy_col] / d[batch_col])

    def _agg_group(g):
        return g.groupby(freq_col, as_index=False).agg(_tp=(throughput_col, 'mean'), _epp=('energy_per_prompt', 'mean')).rename(columns={'_tp': 'throughput', '_epp': 'energy_per_prompt'}).sort_values(freq_col)
    if hue is None:
        series_list = [(None, _agg_group(d))]
    else:
        series_list = [(key, _agg_group(g)) for key, g in d.groupby(hue, sort=False)]
    n = len(series_list)
    fig, ax_left = plt.subplots(figsize=_figsize(350, height=height))
    ax_right = ax_left.twinx()
    ax_right.spines['right'].set_visible(True)
    for i, (key, agg) in enumerate(series_list):
        color = plot_color(i)
        x = agg[freq_col]
        label_tp = f'{key} — throughput' if key is not None else 'Throughput'
        label_ep = f'{key} — energy/prompt' if key is not None else 'Energy / prompt'
        ax_left.plot(x, agg['throughput'], color=color, marker='o', label=label_tp)
        ax_right.plot(x, agg['energy_per_prompt'], color=color, linestyle='--', marker='s', fillstyle='none', label=label_ep)
    ax_left.set_xlabel(freq_label)
    ax_left.set_ylabel(throughput_label)
    ax_right.set_ylabel(energy_per_prompt_label)
    ax_left.set_title(title or 'Throughput and energy per prompt vs frequency')
    h1, l1 = ax_left.get_legend_handles_labels()
    h2, l2 = ax_right.get_legend_handles_labels()
    _legend(ax_left, h1 + h2, l1 + l2, loc='best', frameon=True)
    fig.tight_layout()
    if show:
        plt.show()
    return (fig, ax_left, ax_right)

def plot_time_freq(df, ax=None, max_x=1000, height=None, **kwargs):
    if ax is None:
        w, h = _figsize(350, height=height)
        plt.figure(figsize=(w, h))
        ax = plt.gca()
    pts = df['energy_samples_series'].iloc[0] if hasattr(df['energy_samples_series'], 'iloc') else df['energy_samples_series']
    points = pts.get('0', pts) if isinstance(pts, dict) else pts
    t0 = points[0][0]
    time = [p[0] - t0 for p in points]
    power = [p[1] for p in points]
    volt = [v[2] for v in points]
    freq = [f[3] for f in points]
    ax2 = ax.twinx()
    line1 = sns.lineplot(ax=ax2, x=time, y=freq, marker='s', markersize=4, markevery=10, color=style_cmap(0.2), label='Frequency')
    line2 = sns.lineplot(ax=ax, x=time, y=power, marker='o', markersize=4, markevery=10, color=style_cmap(0.8), label='Power')
    energy = int(df['energy_joules'].iloc[0] / 1000)
    model_name = os.path.basename(df['model'].iloc[0])
    ax.set_ylim(70, 400)
    ax2.set_ylim(700, 2000)
    ax.set_xlabel('Request Index')
    ax.set_ylabel('Power (W)')
    ax2.set_ylabel('Freq (MHz)')
    ax.set_xlim(0, max_x)
    ax.set_title(f'Model {model_name}, Total Energy: {energy} J')
    if ax.get_legend():
        ax.get_legend().remove()
    if ax2.get_legend():
        ax2.get_legend().remove()
    plt.gcf().autofmt_xdate()
    ax.grid(True)

def _series_from_cell(cell):
    if isinstance(cell, dict) and '0' in cell and isinstance(cell['0'], list):
        return cell['0']
    raise ValueError("energy_samples_series cell must be a dict with key '0' and a list value")


def _kv_samples_from_cell(cell):
    if isinstance(cell, list):
        return cell
    if isinstance(cell, dict) and isinstance(cell.get('kv_samples_series'), list):
        return cell['kv_samples_series']
    raise ValueError('kv_samples_series cell must be a list of samples')

def _energy_trajectory(samples):
    if len(samples) < 2:
        times = np.array([row[0] for row in samples], dtype=float)
        power = np.array([row[1] for row in samples], dtype=float)
        freqs = np.array([row[3] for row in samples], dtype=int)
        energy = np.zeros_like(power)
        rel_times = times - times[0] if len(times) else times
        return (rel_times, power, energy, freqs)
    times = np.array([row[0] for row in samples], dtype=float)
    freqs = np.array([row[3] for row in samples], dtype=int)
    power = np.array([row[1] for row in samples], dtype=float)
    rel_times = times - times[0]
    energy = np.zeros_like(power)
    for i in range(1, len(power)):
        dt = rel_times[i] - rel_times[i - 1]
        if dt < 0:
            raise ValueError('timestamps must be strictly increasing')
        energy[i] = energy[i - 1] + power[i - 1] * dt
    return (rel_times, power, energy, freqs)

def plot_energy_power(df, title=None, max_x=100, height=None):
    if len(df) != 1:
        raise ValueError('df must have exactly one row')
    row = df.iloc[0]
    samples = _series_from_cell(row['energy_samples_series'])
    times, power, energy, freqs = _energy_trajectory(samples)
    plt.figure(figsize=_figsize(350, height=height))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    total_energy = round(row['energy_joules'] / 1000, 2)
    plt.suptitle(title or f'Total Energy: {total_energy}KJ')
    sns.lineplot(ax=ax1, x=times, y=power, color=style_cmap(0.8), label='Power')
    ax1.set_ylabel('Power (W)')
    ax1.grid(False)
    sns.lineplot(ax=ax2, x=times, y=freqs, color=style_cmap(0.2), label='Frequency')
    ax2.set_ylabel('Freq (MHz)')
    plt.xlabel('Time (s)')
    ax1.set_ylim(70, 400)
    ax2.set_ylim(700, 2000)
    ax1.set_xlim(0, max_x)
    plt.tight_layout()
    if ax1.get_legend():
        ax1.get_legend().remove()
    if ax2.get_legend():
        ax2.get_legend().remove()
    plt.show()


def plot_kv_cache_utilization_over_time(
    df,
    *,
    kv_series_col='kv_samples_series',
    lane=None,
    util_key='current_utilization',
    agg='mean',
    title=None,
    show=True,
    height=None,
):
    required = {kv_series_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f'DataFrame missing columns: {missing}')
    if df.empty:
        raise ValueError('df is empty')
    if '_result_json_path' in df.columns:
        unique_paths = pd.Series(df['_result_json_path']).dropna().unique()
        if len(unique_paths) > 1:
            raise ValueError('df contains multiple runs; filter to a single result_json path')

    row = df.iloc[0]
    samples = _kv_samples_from_cell(row[kv_series_col])
    if not samples:
        raise ValueError('kv_samples_series is empty')

    if isinstance(lane, (list, tuple, set)):
        lane_ids = [str(item) for item in lane]
    elif lane is None:
        lane_ids = None
    else:
        lane_text = str(lane)
        if lane_text.lower() == 'all':
            lane_ids = 'all'
        else:
            lane_ids = [lane_text]

    if lane_ids == 'all':
        lane_set = set()
        for sample in samples:
            if isinstance(sample, dict):
                lane_set.update((sample.get('per_lane') or {}).keys())
        lane_ids = sorted(lane_set)
        if not lane_ids:
            raise ValueError('kv_samples_series has no lane data')

    agg_name = str(agg or 'mean').strip().lower()
    allowed_aggs = {'mean', 'median', 'min', 'max'}
    if lane_ids is None and agg_name not in allowed_aggs:
        raise ValueError(f'agg must be one of {sorted(allowed_aggs)}, got {agg!r}')

    rows = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        timestamp = sample.get('timestamp')
        per_lane = sample.get('per_lane') or {}
        if lane_ids is None:
            values = []
            for lane_values in per_lane.values():
                if not isinstance(lane_values, dict):
                    continue
                raw = lane_values.get(util_key)
                if raw is None:
                    continue
                try:
                    val = float(raw)
                except (TypeError, ValueError):
                    continue
                if np.isnan(val) or np.isinf(val):
                    continue
                values.append(val)
            if values:
                if agg_name == 'mean':
                    util_value = float(np.mean(values))
                elif agg_name == 'median':
                    util_value = float(np.median(values))
                elif agg_name == 'min':
                    util_value = float(np.min(values))
                else:
                    util_value = float(np.max(values))
            else:
                util_value = None
            rows.append({'timestamp': timestamp, 'utilization': util_value})
        else:
            row = {'timestamp': timestamp}
            for lane_id in lane_ids:
                lane_values = per_lane.get(lane_id) or {}
                row[lane_id] = lane_values.get(util_key)
            rows.append(row)

    plot_df = pd.DataFrame(rows)
    plot_df['timestamp'] = pd.to_numeric(plot_df['timestamp'], errors='coerce')
    plot_df = plot_df.dropna(subset=['timestamp'])
    if plot_df.empty:
        raise ValueError('kv_samples_series has no valid timestamps')
    plot_df = plot_df.sort_values('timestamp')
    t0 = plot_df['timestamp'].iloc[0]
    plot_df['time_s'] = plot_df['timestamp'] - t0

    fig, ax = plt.subplots(figsize=_figsize(350, height=height))
    if lane_ids is None:
        plot_df['utilization'] = pd.to_numeric(plot_df['utilization'], errors='coerce')
        sub = plot_df.dropna(subset=['utilization'])
        if sub.empty:
            raise ValueError('kv_samples_series has no utilization values')
        sns.lineplot(
            ax=ax,
            data=sub,
            x='time_s',
            y='utilization',
            color=plot_color(0),
            linewidth=1.6,
            label=f'{agg_name} utilization',
        )
    else:
        plotted = 0
        for i, lane_id in enumerate(lane_ids):
            if lane_id not in plot_df.columns:
                continue
            series = pd.to_numeric(plot_df[lane_id], errors='coerce')
            sub = plot_df.loc[series.notna(), ['time_s', lane_id]]
            if sub.empty:
                continue
            sns.lineplot(
                ax=ax,
                data=sub,
                x='time_s',
                y=lane_id,
                color=plot_color(i),
                linewidth=1.3,
                label=f'lane {lane_id}',
            )
            plotted += 1
        if plotted == 0:
            raise ValueError('kv_samples_series has no utilization values for selected lanes')

    ax.set_xlabel('Request Index')
    ax.set_ylabel('KV cache utilization')
    ax.set_title(title or 'KV cache utilization over time')
    tick_style(ax)
    if ax.get_legend():
        if lane_ids is None:
            global_legend(ax=ax, ncol=1)
        else:
            global_legend(ax=ax, ncol=min(4, len(lane_ids)))
    plt.tight_layout()
    if show:
        plt.show()
    return (fig, ax)

def plot_epp_sweep_bs_3d(df, ilen=10, olen=1000, bss=None, height=None):
    if bss is None:
        bss = [1, 5, 10, 50, 100, 200]
    freqs = [800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800]
    models = df['model'].unique()
    for model in models:
        fig = plt.figure(figsize=_figsize(700, height=height))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f'Model: {model}')
        x_list, y_list, z_list = ([], [], [])
        for bs in bss:
            for freq in freqs:
                filtered_df = df[(df['batch_size'] == str(bs)) & (df['model'] == model) & (df['input_size'] == str(ilen)) & (df['output_size'] == str(olen)) & (df['frequency'] == str(freq))]
                if not filtered_df.empty:
                    energy_joules = filtered_df['energy_joules'].iloc[0]
                    x_list.append(bs)
                    y_list.append(freq)
                    z_list.append(energy_joules / 200)
        x = np.array(x_list)
        y = np.array(y_list)
        z = np.array(z_list)
        if len(x) == 0:
            plt.show()
            continue
        xi = np.linspace(x.min(), x.max(), 100)
        yi = np.linspace(y.min(), y.max(), 100)
        XI, YI = np.meshgrid(xi, yi)
        ZI = interpolate.griddata((x, y), z, (XI, YI), method='cubic')
        surf = ax.plot_surface(XI, YI, ZI, cmap=style_cmap.reversed(), edgecolor='none', alpha=0.9)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Energy per Prompt')
        ax.view_init(elev=30, azim=-75)
        ax.set_xlabel(CONCURRENT_REQUESTS_LABEL)
        ax.set_ylabel('Frequency')
        ax.set_zlabel('Energy per Prompt')
        plt.show()

def plot_epp_fixed_ilen(df, ilen=10, freq=1100, height=None, show=True):
    olens = [10, 100, 1000]
    bss = [1, 2, 5, 10, 20, 50, 100, 200]
    models = df['model'].unique()
    ncols = len(models)
    nrows = 1
    fig, axes = plt.subplots(nrows, ncols, figsize=_figsize(350, height=height))
    if ncols == 1:
        axes = [axes]
    plot_points = []
    for col, model in enumerate(models):
        for i, olen in enumerate(olens):
            rows = []
            for bs in bss:
                filtered_df = df[(df['batch_size'] == str(bs)) & (df['model'] == model) & (df['input_size'] == str(ilen)) & (df['output_size'] == str(olen)) & (df['frequency'] == str(freq))]
                if not filtered_df.empty:
                    energy_joules = filtered_df['energy_joules'].iloc[0]
                    y_val = energy_joules / 200
                    rows.append({'bs': bs, 'Y': y_val})
                    plot_points.append({
                        'model': model,
                        'input_size': ilen,
                        'output_size': olen,
                        'batch_size': bs,
                        'frequency': freq,
                        'energy_per_prompt': y_val,
                    })
            df_plot = pd.DataFrame(rows)
            if not df_plot.empty:
                df_plot = df_plot.sort_values('bs')
                color = plot_color(i)
                sns.lineplot(
                    ax=axes[col],
                    data=df_plot,
                    x='bs',
                    y='Y',
                    label=str(olen),
                    marker='o',
                    color=color,
                )
                line = axes[col].get_lines()[-1]
                line.set_markeredgecolor(color)
                line.set_markeredgewidth(0)
    master_ax = axes[0]
    for ax in axes[1:]:
        ax.sharey(master_ax)
        ax.set_ylabel('')
        ax.set_xlabel(CONCURRENT_REQUESTS_LABEL)
    master_ax.set_yscale('log')
    master_ax.set_xlabel(CONCURRENT_REQUESTS_LABEL)
    master_ax.set_ylabel('Energy Per Prompt (J)')
    handles, labels = master_ax.get_legend_handles_labels()
    for ax in axes:
        if ax.get_legend():
            ax.get_legend().remove()
    if handles:
        _legend(master_ax, handles, labels, loc='upper right', ncol=3, title='Output Tokens')
    print(f'plot_epp_fixed_ilen data points (ilen={ilen}, freq={freq}):')
    print(
        pd.DataFrame(plot_points).to_string(index=False, float_format=lambda v: f'{v:.4f}')
        if plot_points
        else '(no data points)'
    )
    plt.tight_layout()
    if show:
        plt.show()
    ax = axes[0] if ncols == 1 else axes
    return (fig, ax)

def plot_throughput_heatmap(df, batch_col='batch_size', freq_col='frequency', value_col='tokens_per_second', title=None, cmap=None, annotate=True, height=None):
    if cmap is None:
        cmap = style_cmap
    required = {batch_col, freq_col, value_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f'Missing columns: {missing}')
    df_numeric = df.copy()
    for col in required:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
    df_numeric = df_numeric.dropna(subset=list(required))
    pivot = df_numeric.pivot_table(index=batch_col, columns=freq_col, values=value_col, aggfunc='mean').sort_index().sort_index(axis=1)
    fig, ax = plt.subplots(figsize=_figsize(350, height=height))
    sns.heatmap(pivot, ax=ax, cmap=cmap, linewidths=0.5, linecolor='white', annot=annotate, fmt='.1f' if annotate else None)
    ax.invert_yaxis()
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel(CONCURRENT_REQUESTS_LABEL)
    ax.set_title(title or 'Throughput (TPS)')
    fig.tight_layout()
    plt.show()

def plot_latency_heatmap(df, batch_col='batch_size', freq_col='frequency', value_col='elapsed_time', title=None, cmap=None, annotate=True, height=None):
    if cmap is None:
        cmap = style_cmap
    required = {batch_col, freq_col, value_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f'Missing columns: {missing}')
    df_numeric = df.copy()
    for col in required:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
    df_numeric = df_numeric.dropna(subset=list(required))
    pivot = df_numeric.pivot_table(index=batch_col, columns=freq_col, values=value_col, aggfunc='mean').sort_index().sort_index(axis=1)
    fig, ax = plt.subplots(figsize=_figsize(350, height=height))
    sns.heatmap(pivot, ax=ax, cmap=cmap.reversed(), linewidths=0.5, linecolor='white', annot=annotate, fmt='.1f' if annotate else None)
    ax.invert_yaxis()
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel(CONCURRENT_REQUESTS_LABEL)
    ax.set_title(title or 'Completion Time (s)')
    fig.tight_layout()
    plt.show()

def plot_tokens_per_energy_heatmap(df, batch_col='batch_size', freq_col='frequency', tokens_col='total_num_tokens', energy_col='energy_joules', title=None, cmap=None, annotate=True, batch_order=None, freq_order=None, height=None, show=True):
    if cmap is None:
        cmap = tokens_per_energy_heatmap_cmap
    required = {batch_col, freq_col, tokens_col, energy_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f'Missing columns: {missing}')
    df_numeric = df.copy()
    for col in required:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
    df_numeric = df_numeric.dropna(subset=list(required))
    df_numeric = df_numeric[df_numeric[energy_col] > 0]
    df_numeric = df_numeric.assign(tokens_per_energy=df_numeric[tokens_col] / df_numeric[energy_col])
    pivot = df_numeric.pivot_table(index=batch_col, columns=freq_col, values='tokens_per_energy', aggfunc='mean')
    if batch_order is not None:
        pivot = pivot.reindex(batch_order)
    if freq_order is not None:
        pivot = pivot.reindex(columns=freq_order)
    if batch_order is None and freq_order is None:
        pivot = pivot.sort_index().sort_index(axis=1)
    print('plot_tokens_per_energy_heatmap data points:')
    print(pivot.to_string(float_format=lambda v: f'{v:.4f}' if pd.notna(v) else ''))
    fig, ax = plt.subplots(figsize=_figsize(350, height=height))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap=cmap,
        linewidths=0.5,
        linecolor='white',
        annot=annotate,
        fmt='.2f' if annotate else '',
        cbar_kws={'label': title or 'Tokens Per Joule'},
    )
    ax.invert_yaxis()
    ax.set_xlabel('Frequency')
    ax.set_ylabel(CONCURRENT_REQUESTS_LABEL)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax

def plot_power_heatmap(df, batch_col='batch_size', freq_col='frequency', energy_col='energy_joules', time_col='elapsed_time', title=None, cmap=None, annotate=True, batch_order=None, freq_order=None, height=None):
    if cmap is None:
        cmap = style_cmap
    required = {batch_col, freq_col, energy_col, time_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f'Missing columns: {missing}')
    df_numeric = df.copy()
    for col in required:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
    df_numeric = df_numeric.dropna(subset=list(required))
    df_numeric = df_numeric[df_numeric[time_col] > 0]
    df_numeric = df_numeric.assign(power=df_numeric[energy_col] / df_numeric[time_col])
    pivot = df_numeric.pivot_table(index=batch_col, columns=freq_col, values='power', aggfunc='mean')
    if batch_order is not None:
        pivot = pivot.reindex(batch_order)
    if freq_order is not None:
        pivot = pivot.reindex(columns=freq_order)
    if batch_order is None and freq_order is None:
        pivot = pivot.sort_index().sort_index(axis=1)
    fig, ax = plt.subplots(figsize=_figsize(350, height=height))
    sns.heatmap(pivot, ax=ax, cmap=cmap.reversed(), linewidths=0.5, linecolor='white', annot=annotate, fmt='.2f' if annotate else None)
    ax.invert_yaxis()
    ax.set_xlabel('Frequency')
    ax.set_ylabel(CONCURRENT_REQUESTS_LABEL)
    ax.set_title(title or 'Power (J/s)')
    fig.tight_layout()
    plt.show()

def plot_energy_heatmap_by_ratio(df, freq_col='frequency', input_col='input_size', output_col='output_size', energy_col='energy_joules', title=None, cmap=None, annotate=True, freq_order=None, height=None):
    if cmap is None:
        cmap = style_cmap
    required = {freq_col, input_col, output_col, energy_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f'Missing columns: {missing}')
    df_numeric = df.copy()
    for col in required:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
    df_numeric = df_numeric.dropna(subset=list(required))
    df_numeric = df_numeric[df_numeric[output_col] != 0]
    df_numeric = df_numeric.assign(ratio=df_numeric[input_col] / df_numeric[output_col])
    pivot = df_numeric.pivot_table(index='ratio', columns=freq_col, values=energy_col, aggfunc='mean')
    pivot = pivot.sort_index()
    if freq_order is not None:
        pivot = pivot.reindex(columns=freq_order)
    else:
        pivot = pivot.sort_index(axis=1)
    fig, ax = plt.subplots(figsize=_figsize(350, height=height))
    sns.heatmap(pivot, ax=ax, cmap=cmap.reversed(), linewidths=0.5, linecolor='white', annot=annotate, fmt='.1f' if annotate else None, cbar_kws={'label': 'Energy (J)'})
    ax.invert_yaxis()
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Input/Output Token ratio')
    ax.set_title(title or 'Energy (J)')
    fig.tight_layout()
    plt.show()

def plot_energy_heatmap_by_tokens(df, input_col='input_size', output_col='output_size', energy_col='energy_joules', tokens_col='total_num_tokens', title=None, cmap=None, annotate=True, freq_order=None, height=None):
    if cmap is None:
        cmap = style_cmap
    required = {input_col, output_col, energy_col, tokens_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f'Missing columns: {missing}')
    df_numeric = df.copy()
    for col in required:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
    df_numeric = df_numeric.dropna(subset=list(required))
    df_numeric = df_numeric[df_numeric[energy_col] > 0]
    df_numeric = df_numeric.assign(tokens_per_energy=df_numeric[tokens_col] / df_numeric[energy_col])
    pivot = df_numeric.pivot_table(index=input_col, columns=output_col, values='tokens_per_energy', aggfunc='mean').sort_index().sort_index(axis=1)
    fig, ax = plt.subplots(figsize=_figsize(350, height=height))
    sns.heatmap(pivot, ax=ax, cmap=cmap, linewidths=0.5, linecolor='white', annot=annotate, fmt='.1f' if annotate else None)
    ax.invert_yaxis()
    ax.set_xlabel('Output Tokens')
    ax.set_ylabel('Input Token')
    ax.set_title(title or 'Tokens per Energy (TPJ)')
    fig.tight_layout()
    plt.show()

_MOTIVATION_WINNER_METRICS = {
    'total': {
        'aliases': ('total', 'total_energy', 'energy'),
        'cbar': 'Energy (J)',
        'default_title': 'Min total-energy batch/frequency winner',
    },
    'per_prompt': {
        'aliases': ('per_prompt', 'energy_per_prompt', 'epp'),
        'cbar': 'Energy per prompt (J)',
        'default_title': 'Min energy/prompt batch/frequency winner',
    },
    'per_token': {
        'aliases': ('per_token', 'energy_per_token', 'ept'),
        'cbar': 'Energy per token (J)',
        'default_title': 'Min energy/token batch/frequency winner',
    },
}


def _normalize_motivation_winner_metric(cell_metric):
    key = str(cell_metric or 'per_prompt').strip().lower().replace('-', '_')
    for canonical, spec in _MOTIVATION_WINNER_METRICS.items():
        if key in spec['aliases'] or key == canonical:
            return canonical
    allowed = sorted({a for spec in _MOTIVATION_WINNER_METRICS.values() for a in spec['aliases']})
    raise ValueError(f'cell_metric must be one of {allowed}, got {cell_metric!r}')


def _motivation_winner_cell_values(df, metric, energy_col, batch_col, input_col, output_col, tokens_col):
    if metric == 'total':
        return df[energy_col].astype(float)
    if metric == 'per_prompt':
        return df[energy_col].astype(float) / df[batch_col].astype(float)
    if tokens_col in df.columns:
        tokens = pd.to_numeric(df[tokens_col], errors='coerce')
    else:
        tokens = df[batch_col].astype(float) * (df[input_col].astype(float) + df[output_col].astype(float))
    return df[energy_col].astype(float) / tokens.astype(float)


def plot_motivation_winner_map(
    df,
    input_col='input_size',
    output_col='output_size',
    batch_col='batch_size',
    freq_col='frequency',
    energy_col='energy_joules',
    tokens_col='total_num_tokens',
    cell_metric='per_prompt',
    input_range=None,
    output_range=None,
    title=None,
    cmap=None,
    annotate=True,
    show=True,
    height=None,
):
    metric = _normalize_motivation_winner_metric(cell_metric)
    metric_spec = _MOTIVATION_WINNER_METRICS[metric]
    if cmap is None:
        cmap = style_cmap
    required = {input_col, output_col, batch_col, freq_col, energy_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f'Missing columns: {missing}')
    df_numeric = df.copy()
    for col in required:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
    df_numeric = df_numeric.dropna(subset=list(required))
    if metric == 'per_prompt':
        df_numeric = df_numeric[df_numeric[batch_col] > 0]
    elif metric == 'per_token':
        if tokens_col in df_numeric.columns:
            df_numeric[tokens_col] = pd.to_numeric(df_numeric[tokens_col], errors='coerce')
            df_numeric = df_numeric[df_numeric[tokens_col] > 0]
        else:
            df_numeric = df_numeric[df_numeric[batch_col] > 0]
            df_numeric = df_numeric[(df_numeric[input_col] + df_numeric[output_col]) > 0]
    df_numeric = df_numeric.assign(
        cell_value=_motivation_winner_cell_values(
            df_numeric, metric, energy_col, batch_col, input_col, output_col, tokens_col
        )
    )
    df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan).dropna(subset=['cell_value'])
    agg = df_numeric.groupby([input_col, output_col, batch_col, freq_col], as_index=False)['cell_value'].mean()
    winner_idx = agg.groupby([input_col, output_col])['cell_value'].idxmin()
    winners = agg.loc[winner_idx].copy()

    def _config_label(row):
        bs = row[batch_col]
        fq = row[freq_col]
        bs_text = str(int(bs)) if float(bs).is_integer() else str(bs)
        fq_text = str(int(fq)) if float(fq).is_integer() else str(fq)
        return f'{bs_text}|{fq_text}'

    winners['winner_label'] = winners.apply(_config_label, axis=1)
    energy_pivot = winners.pivot(index=output_col, columns=input_col, values='cell_value')
    label_pivot = winners.pivot(index=output_col, columns=input_col, values='winner_label')
    if input_range is not None:
        input_range = [float(v) for v in input_range]
        energy_pivot = energy_pivot.reindex(columns=input_range)
        label_pivot = label_pivot.reindex(columns=input_range)
    else:
        energy_pivot = energy_pivot.sort_index(axis=1)
        label_pivot = label_pivot.sort_index(axis=1)
    if output_range is not None:
        output_range = [float(v) for v in output_range]
        energy_pivot = energy_pivot.reindex(index=output_range)
        label_pivot = label_pivot.reindex(index=output_range)
    else:
        energy_pivot = energy_pivot.sort_index()
        label_pivot = label_pivot.sort_index()
    fig, ax = plt.subplots(figsize=_figsize(350, height=height))
    annot = label_pivot if annotate else False
    sns.heatmap(
        energy_pivot,
        ax=ax,
        cmap=cmap.reversed(),
        linewidths=0.5,
        linecolor='white',
        annot=annot,
        fmt='',
        cbar_kws={'label': metric_spec['cbar']},
        mask=energy_pivot.isna(),
    )
    ax.invert_yaxis()
    ax.set_xlabel('Input size')
    ax.set_ylabel('Output size')
    ax.set_title(title or metric_spec['default_title'])
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax

def plot_energy_heatmap_by_ratio_per_batch(df, batch_col='batch_size', input_col='input_size', output_col='output_size', energy_col='energy_joules', title=None, cmap=None, annotate=True, freq_order=None, height=None):
    if cmap is None:
        cmap = style_cmap
    required = {batch_col, input_col, output_col, energy_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f'Missing columns: {missing}')
    df_numeric = df.copy()
    for col in required:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
    df_numeric = df_numeric.dropna(subset=list(required))
    df_numeric = df_numeric[df_numeric[output_col] != 0]
    df_numeric = df_numeric.assign(ratio=df_numeric[input_col] / df_numeric[output_col])
    pivot = df_numeric.pivot_table(index='ratio', columns=batch_col, values=energy_col, aggfunc='mean').sort_index().sort_index(axis=1)
    fig, ax = plt.subplots(figsize=_figsize(350, height=height))
    sns.heatmap(pivot, ax=ax, cmap=cmap.reversed(), linewidths=0.5, linecolor='white', annot=annotate, fmt='.1f' if annotate else None)
    ax.invert_yaxis()
    ax.set_xlabel(CONCURRENT_REQUESTS_LABEL)
    ax.set_ylabel('Input/Output Token ratio')
    ax.set_title(title or 'Energy (J)')
    fig.tight_layout()
    plt.show()

def plot_tps_per_energy_heatmap(df, batch_col='batch_size', freq_col='frequency', tokens_col='tokens_per_second', energy_col='energy_joules', title=None, cmap=None, annotate=True, batch_order=None, freq_order=None, height=None):
    if cmap is None:
        cmap = style_cmap
    required = {batch_col, freq_col, tokens_col, energy_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f'Missing columns: {missing}')
    df_numeric = df.copy()
    for col in required:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
    df_numeric = df_numeric.dropna(subset=list(required))
    df_numeric = df_numeric[df_numeric[energy_col] > 0]
    df_numeric = df_numeric.assign(tokens_per_energy=df_numeric[tokens_col] / df_numeric[energy_col])
    pivot = df_numeric.pivot_table(index=batch_col, columns=freq_col, values='tokens_per_energy', aggfunc='mean')
    if batch_order is not None:
        pivot = pivot.reindex(batch_order)
    if freq_order is not None:
        pivot = pivot.reindex(columns=freq_order)
    if batch_order is None and freq_order is None:
        pivot = pivot.sort_index().sort_index(axis=1)
    fig, ax = plt.subplots(figsize=_figsize(350, height=height))
    sns.heatmap(pivot, ax=ax, cmap=cmap, linewidths=0.5, linecolor='white', annot=annotate, fmt='.2f' if annotate else None)
    ax.invert_yaxis()
    ax.set_xlabel('Frequency')
    ax.set_ylabel(CONCURRENT_REQUESTS_LABEL)
    ax.set_title(title or 'Energy Efficiency (TPS/J)')
    fig.tight_layout()
    plt.show()

def plot_throughput_per_energy_heatmap_by_ratio_per_batch(df, batch_col='batch_size', input_col='input_size', output_col='output_size', energy_col='energy_joules', title=None, cmap=None, annotate=True, freq_order=None, height=None):
    if cmap is None:
        cmap = style_cmap
    required = {batch_col, input_col, output_col, energy_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f'Missing columns: {missing}')
    df_numeric = df.copy()
    for col in required:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
    df_numeric = df_numeric.dropna(subset=list(required))
    df_numeric = df_numeric[df_numeric[output_col] != 0]
    df_numeric = df_numeric[df_numeric[energy_col] > 0]
    df_numeric = df_numeric.assign(throughput_per_energy=df_numeric[input_col] / df_numeric[energy_col], ratio=df_numeric[input_col] / df_numeric[output_col])
    pivot = df_numeric.pivot_table(index='ratio', columns=batch_col, values='throughput_per_energy', aggfunc='mean').sort_index().sort_index(axis=1)
    fig, ax = plt.subplots(figsize=_figsize(350, height=height))
    sns.heatmap(pivot, ax=ax, cmap=cmap, linewidths=0.5, linecolor='white', annot=annotate, fmt='.2f' if annotate else None)
    ax.invert_yaxis()
    ax.set_xlabel(CONCURRENT_REQUESTS_LABEL)
    ax.set_ylabel('Input/Output Token ratio')
    ax.set_title(title or 'Throughput per Energy (Tokens/J)')
    fig.tight_layout()
    plt.show()

def plot_throughput_per_energy_heatmap_by_ratio_per_freq(df, freq_col='frequency', input_col='input_size', output_col='output_size', energy_col='energy_joules', title=None, cmap=None, annotate=True, freq_order=None, height=None):
    if cmap is None:
        cmap = style_cmap
    required = {freq_col, input_col, output_col, energy_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f'Missing columns: {missing}')
    df_numeric = df.copy()
    for col in required:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
    df_numeric = df_numeric.dropna(subset=list(required))
    df_numeric = df_numeric[df_numeric[output_col] != 0]
    df_numeric = df_numeric[df_numeric[energy_col] > 0]
    df_numeric = df_numeric.assign(throughput_per_energy=df_numeric[input_col] / df_numeric[energy_col], ratio=df_numeric[input_col] / df_numeric[output_col])
    pivot = df_numeric.pivot_table(index='ratio', columns=freq_col, values='throughput_per_energy', aggfunc='mean').sort_index()
    if freq_order is not None:
        pivot = pivot.reindex(columns=freq_order)
    else:
        pivot = pivot.sort_index(axis=1)
    fig, ax = plt.subplots(figsize=_figsize(350, height=height))
    sns.heatmap(pivot, ax=ax, cmap=cmap, linewidths=0.5, linecolor='white', annot=annotate, fmt='.2f' if annotate else None)
    ax.invert_yaxis()
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Input/Output Token ratio')
    ax.set_title(title or 'Throughput per Energy (Tokens/J)')
    fig.tight_layout()
    plt.show()

def plot_epp_per_prompt_heatmap(df, batch_col='batch_size', freq_col='frequency', energy_col='energy_joules', ilen=18, olen=1000, title=None, cmap=None, annotate=False, batch_order=None, freq_order=None, height=None):
    if cmap is None:
        cmap = style_cmap
    df_filtered = df[(df['input_size'] == str(ilen)) & (df['output_size'] == str(olen))].copy()
    df_filtered[batch_col] = pd.to_numeric(df_filtered[batch_col], errors='coerce')
    df_filtered[freq_col] = pd.to_numeric(df_filtered[freq_col], errors='coerce')
    df_filtered[energy_col] = pd.to_numeric(df_filtered[energy_col], errors='coerce')
    df_filtered = df_filtered.dropna(subset=[batch_col, freq_col, energy_col])
    df_filtered = df_filtered.assign(epp=df_filtered[energy_col] / 298)
    pivot = df_filtered.pivot_table(index=batch_col, columns=freq_col, values='epp', aggfunc='mean')
    if batch_order is not None:
        pivot = pivot.reindex(batch_order)
    if freq_order is not None:
        pivot = pivot.reindex(columns=freq_order)
    if batch_order is None and freq_order is None:
        pivot = pivot.sort_index().sort_index(axis=1)
    fig, ax = plt.subplots(figsize=_figsize(350, height=height))
    sns.heatmap(pivot, ax=ax, cmap=cmap.reversed(), linewidths=0.5, linecolor='white', annot=annotate, fmt='.2f' if annotate else None, cbar_kws={'label': 'Energy per Prompt'})
    ax.invert_yaxis()
    ax.set_xlabel('Frequency')
    ax.set_ylabel(CONCURRENT_REQUESTS_LABEL)
    ax.set_title(title or f'Energy per Prompt (Input:{ilen} Output:{olen})')
    fig.tight_layout()
    plt.show()

def _plot_input_output_scatter_panels(df, panels, input_col, output_col, title='', show=True, show_regression_line=False, xlabel='Input Tokens', ylabel='Output Tokens', height=None):
    w, h = _figsize(350, height=height)
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(w * n, h), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (mask, panel_title, color) in zip(axes, panels):
        sub = df.loc[mask]
        if sub.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        else:
            ax.scatter(sub[input_col], sub[output_col], alpha=0.5, color=color, s=16)
            if show_regression_line and len(sub) >= 2:
                x = sub[input_col].to_numpy(dtype=float)
                y = sub[output_col].to_numpy(dtype=float)
                if np.ptp(x) > 1e-12:
                    coef = np.polyfit(x, y, 1)
                    xx = np.linspace(x.min(), x.max(), 100)
                    ax.plot(xx, np.poly1d(coef)(xx), color=style_cmap(0), linewidth=1.2, linestyle='--', zorder=5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(panel_title)
    xmax = float(df[input_col].max())
    ymax = float(df[output_col].max())
    xpad = max(xmax * 0.02, 1.0)
    ypad = max(ymax * 0.02, 1.0)
    for ax in axes:
        ax.set_xlim(-xpad, xmax + xpad)
        ax.set_ylim(-ypad, ymax + ypad)
    if title:
        fig.suptitle(title, y=1.02)
    plt.tight_layout()
    if show:
        plt.show()
    return (fig, axes)

def plot_token_dist(df, title='', show=True, show_regression_line=False, height=None):
    required = {'ContextTokens', 'GeneratedTokens', 'type'}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f'DataFrame missing columns: {missing}')
    if df.empty:
        raise ValueError('DataFrame is empty')
    panels = [(df['type'] == 0, 'Conversation (type 0)', style_cmap(0.12)), (df['type'] == 1, 'Code (type 1)', style_cmap(0.38)), (df['type'] == 2, 'Math (type 2)', style_cmap(0.62)), (pd.Series(True, index=df.index), 'All', style_cmap(0.9))]
    return _plot_input_output_scatter_panels(df, panels, 'ContextTokens', 'GeneratedTokens', title=title, show=show, show_regression_line=show_regression_line, height=height)

def plot_qps_over_time_per_type(qps_df, *, timestamp_col='timestamp_sec', qps_col='qps', type_col='type', type_labels=None, title=None, show=True, height=None):
    required = {timestamp_col, qps_col, type_col}
    missing = required - set(qps_df.columns)
    if missing:
        raise KeyError(f'qps_df missing columns: {missing}')
    if qps_df.empty:
        raise ValueError('qps_df is empty')
    d = qps_df.copy()
    d[timestamp_col] = pd.to_numeric(d[timestamp_col], errors='coerce')
    d[qps_col] = pd.to_numeric(d[qps_col], errors='coerce')
    d[type_col] = pd.to_numeric(d[type_col], errors='coerce')
    d = d.dropna(subset=[timestamp_col, qps_col, type_col])
    if d.empty:
        raise ValueError('qps_df has no valid numeric rows after coercion')
    d[type_col] = d[type_col].astype(int)

    def _label_for_type(t):
        if type_labels is None:
            return f'type {t}'
        if isinstance(type_labels, dict):
            return type_labels.get(int(t), f'type {t}')
        try:
            return type_labels[int(t)]
        except Exception:
            return f'type {t}'
    types_sorted = sorted(pd.unique(d[type_col]).tolist())
    n = len(types_sorted)
    fig, ax = plt.subplots(figsize=_figsize(350, height=height))
    for i, t in enumerate(types_sorted):
        sub = d[d[type_col] == t].sort_values(timestamp_col)
        color = plot_color(i)
        sns.lineplot(ax=ax, data=sub, x=timestamp_col, y=qps_col, label=_label_for_type(int(t)), color=color, linewidth=1.6)
    ax.set_xlabel('Request Index')
    ax.set_ylabel('QPS')
    ax.set_title(title or 'QPS over time')
    tick_style(ax)
    global_legend(ax=ax, ncol=min(4, max(1, n)))
    plt.tight_layout()
    if show:
        plt.show()
    return (fig, ax)


def plot_ttft_over_time(
    df,
    *,
    ttft_col='ttft_s',
    time_col=None,
    acceptable_ttft=None,
    tail_slo=False,
    title=None,
    show=True,
    height=None,
    top=False,
    bottom=False,
):
    required = {ttft_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f'DataFrame missing columns: {missing}')
    if df.empty:
        raise ValueError('df is empty')
    if '_result_json_path' in df.columns:
        unique_paths = pd.Series(df['_result_json_path']).dropna().unique()
        if len(unique_paths) > 1:
            raise ValueError('df contains multiple runs; filter to a single result_json path')

    d = df.copy()
    if time_col is None:
        for candidate in ('timestamp', 'timestamp_sec', 'arrival_sec', 'time_s', 'prompt_idx'):
            if candidate in d.columns:
                time_col = candidate
                break
    if time_col is None:
        d = d.reset_index(drop=True)
        d['_time_index'] = np.arange(len(d), dtype=float)
        time_col = '_time_index'

    d[time_col] = pd.to_numeric(d[time_col], errors='coerce')
    d[ttft_col] = pd.to_numeric(d[ttft_col], errors='coerce')
    d = d.dropna(subset=[time_col, ttft_col])
    if d.empty:
        raise ValueError('df has no valid ttft/time rows after coercion')
    d = d.sort_values(time_col)
    t0 = d[time_col].iloc[0]
    d = d.assign(time_rel=d[time_col] - t0)

    time_label = 'Time (s)'
    if time_col in ('prompt_idx', '_time_index'):
        time_label = 'Request index'
    elif 'time' not in str(time_col).lower() and not str(time_col).lower().endswith('_s'):
        time_label = str(time_col)

    fig, ax = plt.subplots(figsize=_figsize(350, height=height))
    annotation_index = 0
    def _draw_threshold_line(value, color):
        nonlocal annotation_index
        ax.axhline(
            value,
            color=color,
            linestyle='--',
            linewidth=1.2,
            label='Acceptable TTFT',
        )
        count = int((d[ttft_col] > value).sum())
        total = len(d)
        percent = (count / total * 100) if total else 0.0
        y_pos = 0.5 - (annotation_index * 0.06)
        annotation_index += 1
        ax.text(
            0.98,
            y_pos,
            f'Missed SLO: {percent:.1f}%',
            color='black',
            ha='right',
            va='top',
            transform=ax.transAxes,
            fontsize=max(int(plt.rcParams.get('axes.titlesize', 10)) - 2, 8),
        )
    sns.lineplot(
        ax=ax,
        data=d,
        x='time_rel',
        y=ttft_col,
        color=plot_color(0),
        linewidth=1.6,
        estimator=None,
        label='TTFT',
    )
    if acceptable_ttft is not None:
        try:
            threshold = float(acceptable_ttft)
        except (TypeError, ValueError):
            raise ValueError(f'acceptable_ttft must be numeric, got {acceptable_ttft!r}')
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError(f'acceptable_ttft must be finite and > 0, got {acceptable_ttft!r}')
        _draw_threshold_line(threshold, plot_color(1))
    tail_multiplier = None
    if tail_slo is not False:
        if not (isinstance(tail_slo, tuple) and len(tail_slo) == 2):
            raise ValueError("tail_slo must be a tuple (enabled, multiplier)")
        enabled, multiplier = tail_slo
        if bool(enabled):
            try:
                tail_multiplier = float(multiplier)
            except (TypeError, ValueError):
                raise ValueError("tail_slo multiplier must be numeric")
            if not np.isfinite(tail_multiplier) or tail_multiplier <= 0:
                raise ValueError("tail_slo multiplier must be a positive finite number")
    if tail_multiplier is not None:
        slo = compute_tail_slo_threshold(d, ttft_col=ttft_col, multiplier=tail_multiplier)
        _draw_threshold_line(slo, plot_color(2))

    if top and bottom:
        raise ValueError('top and bottom cannot both be True')
    if bottom:
        ax.set_xlabel('Request Index')
    else:
        ax.set_xlabel('')
    ax.set_ylabel('TTFT (s)')
    ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
    if title:
        ax.text(
            0.5,
            0.98,
            str(title),
            transform=ax.transAxes,
            ha='center',
            va='top',
            fontsize=plt.rcParams.get('axes.titlesize', 10),
        )
    tick_style(ax)
    if ax.get_legend() and top:
        _legend(ax, loc='center left')
    elif ax.get_legend():
        ax.get_legend().remove()
    plt.tight_layout()
    if show:
        plt.show()
    return (fig, ax)


def compute_tail_slo_threshold(df, *, ttft_col='ttft_s', multiplier=5.0):
    """Return (mean ttft) * multiplier for the provided DataFrame."""
    if ttft_col not in df.columns:
        raise KeyError(f'ttft column {ttft_col!r} not present in DataFrame')
    values = pd.to_numeric(df[ttft_col], errors='coerce').dropna()
    if values.empty:
        raise ValueError(f'No valid numeric values in column {ttft_col!r}')
    mean_ttft = float(values.mean())
    if not np.isfinite(mean_ttft):
        raise ValueError('ttft values are not finite, cannot compute tail SLO')
    m = float(multiplier)
    if not np.isfinite(m) or m <= 0:
        raise ValueError('multiplier must be a positive finite number')
    return mean_ttft * m


def _legend_position_params(position):
    if position is None:
        return None
    normalized = str(position).strip().lower()
    mapping = {
        'top left': ('upper left', (0.0, 1.0)),
        'center left': ('center left', (0.0, 0.5)),
        'bottom left': ('lower left', (0.0, 0.0)),
        'top right': ('upper right', (1.0, 1.0)),
        'center right': ('center right', (1.0, 0.5)),
        'bottom right': ('lower right', (1.0, 0.0)),
    }
    if normalized not in mapping:
        raise ValueError(
            f"legend_position must be one of {sorted(mapping.keys())}, got {position!r}"
        )
    loc, anchor = mapping[normalized]
    return {'loc': loc, 'bbox_to_anchor': anchor}


def plot_tpot_over_time(
    df,
    *,
    tpot_col='tpot_mean_s',
    tail_col='tpot_99.9th_s',
    time_col=None,
    acceptable_tpot=None,
    tail_slo=False,
    title=None,
    show=True,
    height=None,
    top=False,
    bottom=False,
): 
    required = {tpot_col, tail_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f'DataFrame missing columns: {missing}')
    if df.empty:
        raise ValueError('df is empty')
    if '_result_json_path' in df.columns:
        unique_paths = pd.Series(df['_result_json_path']).dropna().unique()
        if len(unique_paths) > 1:
            raise ValueError('df contains multiple runs; filter to a single result_json path')
    d = df.copy()
    if time_col is None:
        for candidate in ('timestamp', 'timestamp_sec', 'arrival_sec', 'time_s', 'prompt_idx'):
            if candidate in d.columns:
                time_col = candidate
                break
    if time_col is None:
        d = d.reset_index(drop=True)
        d['_time_index'] = np.arange(len(d), dtype=float)
        time_col = '_time_index'
    d[time_col] = pd.to_numeric(d[time_col], errors='coerce')
    d[tpot_col] = pd.to_numeric(d[tpot_col], errors='coerce')
    d[tail_col] = pd.to_numeric(d[tail_col], errors='coerce')
    d = d.dropna(subset=[time_col, tpot_col, tail_col])
    if d.empty:
        raise ValueError('df has no valid tpot/time rows after coercion')
    d = d.sort_values(time_col)
    t0 = d[time_col].iloc[0]
    d = d.assign(time_rel=d[time_col] - t0)
    time_label = 'Time (s)'
    if time_col in ('prompt_idx', '_time_index'):
        time_label = 'Request index'
    elif 'time' not in str(time_col).lower() and not str(time_col).lower().endswith('_s'):
        time_label = str(time_col)
    fig, ax = plt.subplots(figsize=_figsize(350, height=height))
    annotation_index = 0
    def _draw_threshold_line(value, color):
        nonlocal annotation_index
        ax.axhline(
            value,
            color=color,
            linestyle='--',
            linewidth=1.2,
            label='Acceptable TPOT',
        )
        count = int((d[tpot_col] > value).sum())
        total = len(d)
        percent = (count / total * 100) if total else 0.0
        y_pos = 0.5 - (annotation_index * 0.06)
        annotation_index += 1
        ax.text(
            0.98,
            y_pos,
            f'Missed SLO: {percent:.1f}%',
            color='black',
            ha='right',
            va='top',
            transform=ax.transAxes,
            fontsize=max(int(plt.rcParams.get('axes.titlesize', 10)) - 2, 8),
        )
    sns.lineplot(
        ax=ax,
        data=d,
        x='time_rel',
        y=tpot_col,
        color=plot_color(0),
        linewidth=1.6,
        estimator=None,
        label='TPOT mean',
    )
    sns.lineplot(
        ax=ax,
        data=d,
        x='time_rel',
        y=tail_col,
        color=plot_color(1),
        linewidth=1.6,
        estimator=None,
        label='TPOT 99.9th',
    )
    if acceptable_tpot is not None:
        try:
            threshold = float(acceptable_tpot)
        except (TypeError, ValueError):
            raise ValueError(f'acceptable_tpot must be numeric, got {acceptable_tpot!r}')
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError(f'acceptable_tpot must be finite and > 0, got {acceptable_tpot!r}')
        _draw_threshold_line(threshold, plot_color(2))
    tail_multiplier = None
    if tail_slo is not False:
        if not (isinstance(tail_slo, tuple) and len(tail_slo) == 2):
            raise ValueError("tail_slo must be a tuple (enabled, multiplier)")
        enabled, multiplier = tail_slo
        if bool(enabled):
            try:
                tail_multiplier = float(multiplier)
            except (TypeError, ValueError):
                raise ValueError("tail_slo multiplier must be numeric")
            if not np.isfinite(tail_multiplier) or tail_multiplier <= 0:
                raise ValueError("tail_slo multiplier must be a positive finite number")
    if tail_multiplier is not None:
        slo = compute_tail_slo_threshold(d, ttft_col=tpot_col, multiplier=tail_multiplier)
        _draw_threshold_line(slo, plot_color(3))
    if top and bottom:
        raise ValueError('top and bottom cannot both be True')
    if bottom:
        ax.set_xlabel('Request Index')
    else:
        ax.set_xlabel('')
    ax.set_ylabel('TPOT (s)')
    ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
    if title:
        ax.text(
            0.5,
            0.98,
            str(title),
            transform=ax.transAxes,
            ha='center',
            va='top',
            fontsize=plt.rcParams.get('axes.titlesize', 10),
        )
    tick_style(ax)
    if ax.get_legend() and top:
        _legend(ax, loc='center left')
    elif ax.get_legend():
        ax.get_legend().remove()
    plt.tight_layout()
    if show:
        plt.show()
    return (fig, ax)


def plot_tpot_over_time_compact(
    dataframes,
    labels,
    *,
    acceptable_tpot,
    tpot_col='tpot_mean_s',
    time_col=None,
    title=None,
    show=True,
    height=None,
    legend_position='bottom left',
    width=1,
    print_points=False,
):
    if acceptable_tpot is None:
        raise ValueError('acceptable_tpot must be provided')
    try:
        threshold = float(acceptable_tpot)
    except (TypeError, ValueError):
        raise ValueError(f'acceptable_tpot must be numeric, got {acceptable_tpot!r}')
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError(f'acceptable_tpot must be finite and > 0, got {acceptable_tpot!r}')

    frames = list(dataframes or [])
    if not frames:
        raise ValueError('dataframes must be a non-empty list')
    if labels is None:
        raise ValueError('labels must be provided')
    if len(labels) != len(frames):
        raise ValueError('labels length must match dataframes length')

    fig, ax = plt.subplots(figsize=_figsize(350 * float(width), height=height))
    for idx, (df, label) in enumerate(zip(frames, labels)):
        if tpot_col not in df.columns:
            raise KeyError(f'DataFrame missing column: {tpot_col!r}')
        if df.empty:
            continue
        d = df.copy()
        local_time_col = time_col
        if local_time_col is None:
            for candidate in ('timestamp', 'timestamp_sec', 'arrival_sec', 'time_s', 'prompt_idx'):
                if candidate in d.columns:
                    local_time_col = candidate
                    break
        if local_time_col is None:
            d = d.reset_index(drop=True)
            d['_time_index'] = np.arange(len(d), dtype=float)
            local_time_col = '_time_index'
        d[local_time_col] = pd.to_numeric(d[local_time_col], errors='coerce')
        d[tpot_col] = pd.to_numeric(d[tpot_col], errors='coerce')
        d = d.dropna(subset=[local_time_col, tpot_col])
        if d.empty:
            continue
        d = d.sort_values(local_time_col)
        t0 = d[local_time_col].iloc[0]
        d = d.assign(time_rel=d[local_time_col] - t0)
        missed = int((d[tpot_col] > threshold).sum())
        total = len(d)
        percent = (missed / total * 100) if total else 0.0
        display_label = f'{label} ({percent:.1f}\% Missed)'
        if print_points:
            print(f"{label} datapoints:\n{d[['time_rel', tpot_col]].to_string(index=False)}")
        sns.lineplot(
            ax=ax,
            data=d,
            x='time_rel',
            y=tpot_col,
            color=plot_color(idx),
            linewidth=1.6,
            estimator=None,
            label=display_label,
        )

    ax.axhline(
        threshold,
        color=plot_color(4),
        linestyle='--',
        linewidth=1.2,
        label='Acceptable SLO',
    )
    ax.set_xlabel('Request Index')
    ax.set_ylabel('TPOT (s)')
    ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
    if title:
        ax.text(
            0.5,
            0.98,
            str(title),
            transform=ax.transAxes,
            ha='center',
            va='top',
            fontsize=plt.rcParams.get('axes.titlesize', 10),
        )
    tick_style(ax)
    legend_kwargs = _legend_position_params(legend_position)
    if ax.get_legend() and legend_kwargs:
        _legend(ax, **legend_kwargs)
    elif ax.get_legend():
        ax.get_legend().remove()
    plt.tight_layout()
    if show:
        plt.show()
    return (fig, ax)


def plot_ttft_over_time_compact(
    dataframes,
    labels,
    *,
    acceptable_ttft,
    ttft_col='ttft_s',
    time_col=None,
    title=None,
    show=True,
    height=None,
    legend_position='center left',
    width=1,
    print_points=False,
):
    if acceptable_ttft is None:
        raise ValueError('acceptable_ttft must be provided')
    try:
        threshold = float(acceptable_ttft)
    except (TypeError, ValueError):
        raise ValueError(f'acceptable_ttft must be numeric, got {acceptable_ttft!r}')
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError(f'acceptable_ttft must be finite and > 0, got {acceptable_ttft!r}')

    frames = list(dataframes or [])
    if not frames:
        raise ValueError('dataframes must be a non-empty list')
    if labels is None:
        raise ValueError('labels must be provided')
    if len(labels) != len(frames):
        raise ValueError('labels length must match dataframes length')

    fig, ax = plt.subplots(figsize=_figsize(350 * float(width), height=height))
    for idx, (df, label) in enumerate(zip(frames, labels)):
        if ttft_col not in df.columns:
            raise KeyError(f'DataFrame missing column: {ttft_col!r}')
        if df.empty:
            continue
        d = df.copy()
        local_time_col = time_col
        if local_time_col is None:
            for candidate in ('timestamp', 'timestamp_sec', 'arrival_sec', 'time_s', 'prompt_idx'):
                if candidate in d.columns:
                    local_time_col = candidate
                    break
        if local_time_col is None:
            d = d.reset_index(drop=True)
            d['_time_index'] = np.arange(len(d), dtype=float)
            local_time_col = '_time_index'
        d[local_time_col] = pd.to_numeric(d[local_time_col], errors='coerce')
        d[ttft_col] = pd.to_numeric(d[ttft_col], errors='coerce')
        d = d.dropna(subset=[local_time_col, ttft_col])
        if d.empty:
            continue
        d = d.sort_values(local_time_col)
        t0 = d[local_time_col].iloc[0]
        d = d.assign(time_rel=d[local_time_col] - t0)
        missed = int((d[ttft_col] > threshold).sum())
        total = len(d)
        percent = (missed / total * 100) if total else 0.0
        display_label = f'{label} ({percent:.1f}\% Missed)'
        if print_points:
            print(f"{label} datapoints:\n{d[['time_rel', ttft_col]].to_string(index=False)}")
        sns.lineplot(
            ax=ax,
            data=d,
            x='time_rel',
            y=ttft_col,
            color=plot_color(idx),
            linewidth=1.6,
            estimator=None,
            label=display_label,
        )

    ax.axhline(
        threshold,
        color=plot_color(4),
        linestyle='--',
        linewidth=1.2,
        label='Acceptable SLO',
    )
    ax.set_xlabel('Request Index')
    ax.set_ylabel('TTFT (s)')
    ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
    if title:
        ax.text(
            0.5,
            0.98,
            str(title),
            transform=ax.transAxes,
            ha='center',
            va='top',
            fontsize=plt.rcParams.get('axes.titlesize', 10),
        )
    tick_style(ax)
    legend_kwargs = _legend_position_params(legend_position)
    if ax.get_legend() and legend_kwargs:
        _legend(ax, **legend_kwargs)
    elif ax.get_legend():
        ax.get_legend().remove()
    plt.tight_layout()
    if show:
        plt.show()
    return (fig, ax)


def _load_trace_arrivals_csv(csv_path, *, arrival_col='arrival_sec', group_col='pool'):
    usecols = [arrival_col, group_col]
    df = pd.read_csv(csv_path, usecols=usecols)
    df[arrival_col] = pd.to_numeric(df[arrival_col], errors='coerce')
    df = df.dropna(subset=[arrival_col])
    if df.empty:
        raise ValueError(f'{csv_path!r} has no valid rows in column {arrival_col!r}')
    return df


def _trace_subset_qps(df, *, arrival_col, window_s):
    from dataload import queries_over_time_2s_window

    if df.empty:
        return pd.DataFrame({'timestamp_sec': pd.Series(dtype='int64'), 'qps': pd.Series(dtype='float64')})
    work = df[[arrival_col]].copy()
    work['_panel'] = 0
    qps_df = queries_over_time_2s_window(
        work,
        timestamp_col=arrival_col,
        type_col='_panel',
        expected_types=[0],
        window_s=window_s,
    )
    return qps_df[['timestamp_sec', 'qps']]


def plot_trace_qps_panels(
    csv_path,
    *,
    arrival_col='arrival_sec',
    group_col='type',
    expected_groups=(0, 2, 1),
    group_labels=None,
    window_s=2,
    title=None,
    show=True,
    height=None,
):
    df = _load_trace_arrivals_csv(csv_path, arrival_col=arrival_col, group_col=group_col)
    if group_labels is None:
        group_labels = {
            0: 'Conversation',
            1: 'Code',
            2: 'Summarization',
            'conversation': 'Conversation',
            'summarization': 'Summarization',
            'code': 'Code',
        }

    def _label(group):
        if isinstance(group_labels, dict):
            return group_labels.get(group, str(group))
        return str(group)

    groups = list(expected_groups)
    n = len(groups)
    w, h = _figsize(350, height=height)
    fig, ax = plt.subplots(n, 1, figsize=(w, h), sharex=True)
    ax = np.atleast_1d(ax)
    middle_idx = n // 2
    for i, (panel_ax, group) in enumerate(zip(ax, groups)):
        if group_col == 'type':
            mask = pd.to_numeric(df[group_col], errors='coerce') == int(group)
        else:
            mask = df[group_col].astype(str) == str(group)
        sub = df.loc[mask, [arrival_col]]
        qps_df = _trace_subset_qps(sub, arrival_col=arrival_col, window_s=window_s)
        color = plot_color(0)
        if qps_df.empty:
            panel_ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=panel_ax.transAxes)
        else:
            sns.lineplot(
                ax=panel_ax,
                data=qps_df.sort_values('timestamp_sec'),
                x='timestamp_sec',
                y='qps',
                color=color,
                linewidth=0.8,
            )
        if i == middle_idx:
            panel_ax.set_ylabel('QPS', loc='center')
        else:
            panel_ax.set_ylabel('')
        panel_ax.text(
            0.98,
            0.95,
            _label(group),
            transform=panel_ax.transAxes,
            ha='right',
            va='top',
            fontsize=plt.rcParams.get('axes.titlesize', 10),
        )
        panel_ax.yaxis.set_major_locator(mticker.MaxNLocator(3))
    ax[-1].xaxis.set_major_locator(mticker.MaxNLocator(4))
    ax[-1].set_xlabel('Time (s)')
    if title:
        fig.suptitle(title, y=1.02)
    plt.tight_layout()
    if show:
        plt.show()
    return (fig, ax)


def _draw_workload_token_panel(
    panel_ax,
    values,
    bin_edges,
    *,
    hist_color,
    cdf_color,
    panel_title,
    show_count_ylabel=False,
    show_cdf_ylabel=False,
    log_x=False,
):
    twin = None
    if values is None or len(values) == 0:
        panel_ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=panel_ax.transAxes)
    elif bin_edges is not None:
        panel_ax.hist(
            values,
            bins=bin_edges,
            alpha=0.5,
            color=hist_color,
            edgecolor='black',
            zorder=1,
        )
        twin = panel_ax.twinx()
        x_sorted = np.sort(values)
        count = len(x_sorted)
        y_cdf = np.arange(1, count + 1, dtype=float) / count
        twin.fill_between(
            x_sorted,
            0,
            y_cdf,
            step='post',
            alpha=0.2,
            color=cdf_color,
            zorder=2,
            linewidth=0,
        )
        twin.step(x_sorted, y_cdf, where='post', color=cdf_color, linewidth=0.8, zorder=5)
        twin.set_ylim(0, 1)
        twin.spines['right'].set_visible(True)
        twin.spines['right'].set_color('black')
        twin.spines['right'].set_linewidth(1.5)
        if show_cdf_ylabel:
            twin.set_ylabel('CDF', loc='center')
        else:
            twin.set_ylabel('')
        twin.yaxis.set_major_locator(mticker.MaxNLocator(3))
    if show_count_ylabel:
        panel_ax.set_ylabel('Count', loc='center')
    else:
        panel_ax.set_ylabel('')
    if panel_title:
        panel_ax.text(
            0.98,
            0.95,
            panel_title,
            transform=panel_ax.transAxes,
            ha='right',
            va='top',
            fontsize=plt.rcParams.get('axes.titlesize', 10),
        )
    panel_ax.yaxis.set_major_locator(mticker.MaxNLocator(3))
    if log_x:
        panel_ax.set_xscale('log')
        if twin is not None:
            twin.set_xscale('log')
    return twin


def plot_token_distribution_panels(
    df,
    *,
    title='',
    bins=50,
    show=True,
    type_labels=None,
    expected_types=(0, 2, 1),
    log_x=False,
    height=None,
):
    required = {'ContextTokens', 'GeneratedTokens', 'type'}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f'DataFrame missing columns: {missing}')
    if df.empty:
        raise ValueError('DataFrame is empty')
    if type_labels is None:
        type_labels = {
            0: 'Conversation',
            1: 'Code',
            2: 'Summarization',
        }
    hist_color = plot_color(0)
    cdf_color = plot_color(1)
    types = [int(t) for t in expected_types]
    column_specs = [
        ('ContextTokens', 'Prompt Tokens'),
        ('GeneratedTokens', 'Generated Tokens'),
    ]
    n = len(types)
    middle_idx = n // 2
    w, h = _figsize(350, height=height)
    fig, ax = plt.subplots(n, len(column_specs), figsize=(w * 1.2, h), sharex='col')
    ax = np.atleast_2d(ax)
    cdf_twins = []
    for col_index, (column, xlabel) in enumerate(column_specs):
        vals_all = pd.to_numeric(df[column], errors='coerce').dropna().to_numpy(dtype=float)
        bin_edges = np.histogram_bin_edges(vals_all, bins=bins) if vals_all.size else None
        for row_index, type_val in enumerate(types):
            panel_ax = ax[row_index, col_index]
            type_mask = pd.to_numeric(df['type'], errors='coerce') == type_val
            sub = pd.to_numeric(df.loc[type_mask, column], errors='coerce').dropna()
            values = None if sub.empty else sub.to_numpy(dtype=float)
            is_left = col_index == 0
            is_right = col_index == 1
            is_middle = row_index == middle_idx
            twin = _draw_workload_token_panel(
                panel_ax,
                values,
                bin_edges,
                hist_color=hist_color,
                cdf_color=cdf_color,
                panel_title=type_labels.get(type_val, str(type_val)) if is_right else '',
                show_count_ylabel=is_middle and is_left,
                show_cdf_ylabel=is_middle and not is_left,
                log_x=log_x,
            )
            cdf_twins.append(twin)
        ax[-1, col_index].xaxis.set_major_locator(mticker.MaxNLocator(4))
        ax[-1, col_index].set_xlabel(xlabel)
    active_twins = [t for t in cdf_twins if t is not None]
    if active_twins:
        base_twin = active_twins[0]
        for twin_ax in active_twins[1:]:
            twin_ax.sharey(base_twin)
        base_twin.set_ylim(0, 1)
    if title:
        fig.suptitle(title, y=1.02)
    plt.tight_layout()
    if show:
        plt.show()
    return (fig, ax)


def _plot_seq_token_histogram_panels(df, column, xlabel, title='', show=True, bins=50, height=None):
    required = {'ContextTokens', 'GeneratedTokens', 'type'}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f'DataFrame missing columns: {missing}')
    if df.empty:
        raise ValueError('DataFrame is empty')
    cdf_color = style_cmap(0)
    p50_color = style_cmap(0.25)
    p99_color = style_cmap(0.75)
    w, h = _figsize(350, height=height)
    fig, axes = plt.subplots(1, 4, figsize=(w * 4, h), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    vals_all = df[column].dropna().to_numpy(dtype=float)
    bin_edges = np.histogram_bin_edges(vals_all, bins=bins) if vals_all.size else None
    panels = [(df['type'] == 0, 'Conversation (type 0)', style_cmap(0.12)), (df['type'] == 1, 'Code (type 1)', style_cmap(0.38)), (df['type'] == 2, 'Math (type 2)', style_cmap(0.62)), (pd.Series(True, index=df.index), 'All', style_cmap(0.9))]
    cdf_twins = []
    for ax, (mask, panel_title, color) in zip(axes, panels):
        sub = df.loc[mask, column].dropna()
        twin = None
        if sub.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        elif bin_edges is not None:
            ax.hist(sub.to_numpy(dtype=float), bins=bin_edges, alpha=0.5, color=color, edgecolor='black', zorder=1)
            twin = ax.twinx()
            x_sorted = np.sort(sub.to_numpy(dtype=float))
            n = len(x_sorted)
            y_cdf = np.arange(1, n + 1, dtype=float) / n
            twin.fill_between(x_sorted, 0, y_cdf, step='post', alpha=0.2, color=cdf_color, zorder=2, linewidth=0)
            twin.step(x_sorted, y_cdf, where='post', color=cdf_color, linewidth=1.2, linestyle='-', zorder=5)
            twin.set_ylim(0, 1)
            twin.tick_params(axis='y', labelright=True)
            p50, p99 = np.percentile(sub.to_numpy(dtype=float), [50, 99])
            ax.axvline(p50, color=p50_color, linestyle='--', linewidth=1.2, zorder=7)
            ax.axvline(p99, color=p99_color, linestyle='--', linewidth=1.2, zorder=7)
        cdf_twins.append(twin)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Count')
        ax.set_title(panel_title)
    active_twins = [t for t in cdf_twins if t is not None]
    if active_twins:
        base_twin = active_twins[0]
        for t in active_twins[1:]:
            t.sharey(base_twin)
        base_twin.set_ylim(0, 1)
        active_twins[-1].set_ylabel('CDF')
    legend_handles = [Line2D([0], [0], color=p50_color, linestyle='--', linewidth=1.2), Line2D([0], [0], color=p99_color, linestyle='--', linewidth=1.2)]
    style_legend(fig.legend(legend_handles, ['50th percentile', '99th percentile'], loc='upper center', frameon=True, borderaxespad=0.5, ncol=2, bbox_to_anchor=(0.4, 1.12)))
    if title:
        fig.suptitle(title, y=1.02)
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    if show:
        plt.show()

def plot_input_token_histograms(df, title='', show=True, bins=50, height=None):
    _plot_seq_token_histogram_panels(df, 'ContextTokens', 'Input Tokens', title=title, show=show, bins=bins, height=height)

def plot_output_token_histograms(df, title='', show=True, bins=50, height=None):
    _plot_seq_token_histogram_panels(df, 'GeneratedTokens', 'Output Tokens', title=title, show=show, bins=bins, height=height)

def plot_energy_distribution_and_cdf(df, energy_col='energy_joules', title='Energy distribution', show=True, bins=50, log_x=False, annotate_percentiles=(50, 90, 95, 99), height=None):
    if df is None:
        raise ValueError('df is None')
    if energy_col not in df.columns:
        raise KeyError(f'DataFrame missing column: {energy_col!r}')
    if df.empty:
        raise ValueError('DataFrame is empty')
    vals = df[energy_col].dropna().to_numpy(dtype=float)
    if vals.size == 0:
        raise ValueError(f'No non-NaN values in column {energy_col!r}')
    vals = vals / 1000.0
    fig, ax = plt.subplots(figsize=_figsize(350, height=height))
    color = style_cmap(0.9)
    cdf_color = style_cmap(0)
    bin_edges = np.histogram_bin_edges(vals, bins=bins)
    ax.hist(vals, bins=bin_edges, alpha=0.55, color=color, edgecolor='black', zorder=1)
    ax.set_xlabel('Energy (kJ)')
    ax.set_ylabel('Count')
    ax.set_title(title)
    if log_x:
        ax.set_xscale('log')
    ax_cdf = ax.twinx()
    x_sorted = np.sort(vals)
    n = len(x_sorted)
    y_cdf = np.arange(1, n + 1, dtype=float) / n
    ax_cdf.fill_between(x_sorted, 0, y_cdf, step='post', alpha=0.18, color=cdf_color, zorder=2, linewidth=0)
    ax_cdf.step(x_sorted, y_cdf, where='post', color=cdf_color, linewidth=1.4, linestyle='-', zorder=5)
    ax_cdf.set_ylim(0, 1)
    ax_cdf.set_ylabel('CDF')
    if annotate_percentiles:
        percs = np.asarray(list(annotate_percentiles), dtype=float)
        percs = percs[(percs >= 0) & (percs <= 100)]
        if percs.size:
            xps = np.percentile(vals, percs)
            for p, xp in zip(percs, xps):
                ax.axvline(xp, color=style_cmap(0.25), linestyle='--', linewidth=1.1, zorder=7)
                ax_cdf.text(xp, min(0.98, 0.02 + p / 100.0), f'p{int(p)}', rotation=90, va='bottom', ha='right', fontsize=8, color=style_cmap(0.25))
    plt.tight_layout()
    if show:
        plt.show()
    return (fig, ax, ax_cdf)

def plot_truth_vs_pred(y_true, y_pred, title, height=None):
    fig, ax = plt.subplots(figsize=_figsize(350, height=height))
    ax.scatter(y_true, y_pred, alpha=0.5, color=style_cmap(0.95), s=2)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], '--', color=style_cmap(0))
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    ax.set_title(title)
    plt.tight_layout()

def _scatter_metrics(y_true, y_pred):
    if y_true.size == 0:
        return {'n': 0, 'mae': float('nan'), 'rmse': float('nan'), 'r2': float('nan')}
    diff = y_pred - y_true
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    ss_res = float(np.sum(diff * diff))
    mean_true = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - mean_true) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return {'n': int(y_true.size), 'mae': mae, 'rmse': rmse, 'r2': r2}

def plot_predictor_truth_vs_pred(df, *, title='', true_col='GeneratedTokens', pred_col='PredictedTokens', type_col='type', show=True, type_labels=None, log_scale=False, height=None):
    required = {true_col, pred_col, type_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f'DataFrame missing columns: {missing}')
    if df.empty:
        raise ValueError('DataFrame is empty')
    if type_labels is None:
        type_labels = {0: 'Conversation (type 0)', 1: 'Code (type 1)', 2: 'Summarization (type 2)'}
    work = df[[true_col, pred_col, type_col]].copy()
    work[true_col] = pd.to_numeric(work[true_col], errors='coerce')
    work[pred_col] = pd.to_numeric(work[pred_col], errors='coerce')
    work[type_col] = pd.to_numeric(work[type_col], errors='coerce')
    work = work.dropna(subset=[true_col, pred_col, type_col])
    if work.empty:
        raise ValueError('no numeric rows after coercion')
    work[type_col] = work[type_col].astype(int)
    type_colors = {0: style_cmap(0.12), 1: style_cmap(0.38), 2: style_cmap(0.62)}
    all_color = style_cmap(0.9)
    diag_color = style_cmap(0)
    panel_specs = [(work[type_col] == 0, type_labels.get(0, 'type 0'), type_colors[0]), (work[type_col] == 1, type_labels.get(1, 'type 1'), type_colors[1]), (work[type_col] == 2, type_labels.get(2, 'type 2'), type_colors[2]), (pd.Series(True, index=work.index), 'All', all_color)]
    w, h = _figsize(350, height=height)
    fig, axes = plt.subplots(1, 4, figsize=(w * 4, h), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    global_min = float(min(work[true_col].min(), work[pred_col].min()))
    global_max = float(max(work[true_col].max(), work[pred_col].max()))
    if not np.isfinite(global_min) or not np.isfinite(global_max) or global_min == global_max:
        global_min, global_max = (0.0, 1.0)
    span = global_max - global_min
    pad = max(1.0, 0.05 * span)
    if log_scale:
        lo = max(1.0, global_min)
        hi = max(lo + 1.0, global_max)
        lims = (lo, hi * 1.05)
    else:
        lims = (global_min - pad, global_max + pad)
    for ax, (mask, panel_title, color) in zip(axes, panel_specs):
        sub = work.loc[mask]
        y_true = sub[true_col].to_numpy(dtype=float)
        y_pred = sub[pred_col].to_numpy(dtype=float)
        metrics = _scatter_metrics(y_true, y_pred)
        if y_true.size == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        else:
            ax.scatter(y_true, y_pred, alpha=0.5, color=color, s=6, edgecolor='none')
        ax.plot(lims, lims, '--', color=diag_color, linewidth=1.0)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal', adjustable='box')
        if log_scale:
            ax.set_xscale('log')
            ax.set_yscale('log')
        ax.set_xlabel('True tokens')
        ax.set_ylabel('Predicted tokens')
        if metrics['n'] > 0:
            stats_text = 'n=%d\nMAE=%.1f\nRMSE=%.1f\nR^2=%.3f' % (metrics['n'], metrics['mae'], metrics['rmse'], metrics['r2'])
            ax.text(0.04, 0.96, stats_text, transform=ax.transAxes, ha='left', va='top', fontsize=9, bbox=dict(facecolor='white', edgecolor='black', alpha=0.85, boxstyle='round,pad=0.3'))
        ax.set_title(panel_title)
    if title:
        fig.suptitle(title, y=1.05)
    plt.tight_layout()
    if show:
        plt.show()
    return (fig, axes)

def _ensure_predictor_bucket_columns(df, *, true_col='GeneratedTokens', pred_col='PredictedTokens', true_bucket_col='true_bucket', pred_bucket_col='pred_bucket'):
    from predictors.output_length.buckets import token_length_to_bucket_id
    work = df.copy()
    if true_col not in work.columns or pred_col not in work.columns:
        raise KeyError('need %s and %s for bucket derivation' % (true_col, pred_col))
    if true_bucket_col not in work.columns:
        work[true_bucket_col] = work[true_col].map(token_length_to_bucket_id)
    if pred_bucket_col not in work.columns:
        work[pred_bucket_col] = work[pred_col].map(token_length_to_bucket_id)
    work[true_bucket_col] = pd.to_numeric(work[true_bucket_col], errors='coerce')
    work[pred_bucket_col] = pd.to_numeric(work[pred_bucket_col], errors='coerce')
    return work

def _bucket_confusion_matrix(true_ids, pred_ids, num_buckets):
    matrix = np.zeros((int(num_buckets), int(num_buckets)), dtype=np.int64)
    for true_id, pred_id in zip(true_ids, pred_ids):
        matrix[int(true_id), int(pred_id)] += 1
    return matrix

def _bucket_classification_metrics(true_ids, pred_ids, num_buckets):
    true_ids = np.asarray(true_ids, dtype=np.int64)
    pred_ids = np.asarray(pred_ids, dtype=np.int64)
    n = int(true_ids.size)
    if n == 0:
        return {'n': 0, 'accuracy': float('nan'), 'adjacent_accuracy': float('nan')}
    accuracy = float(np.mean(true_ids == pred_ids))
    adjacent = float(np.mean(np.abs(true_ids - pred_ids) <= 1))
    return {'n': n, 'accuracy': accuracy, 'adjacent_accuracy': adjacent}

def plot_predictor_bucket_classification(df, *, title='', true_col='GeneratedTokens', pred_col='PredictedTokens', type_col='type', true_bucket_col='true_bucket', pred_bucket_col='pred_bucket', show=True, type_labels=None, normalize=True, height=None):
    from predictors.output_length.buckets import BUCKETS, bucket_label
    required = {true_col, pred_col, type_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f'DataFrame missing columns: {missing}')
    if df.empty:
        raise ValueError('DataFrame is empty')
    has_classifier_buckets = true_bucket_col in df.columns and pred_bucket_col in df.columns
    work = _ensure_predictor_bucket_columns(df, true_col=true_col, pred_col=pred_col, true_bucket_col=true_bucket_col, pred_bucket_col=pred_bucket_col)
    work = work.dropna(subset=[true_bucket_col, pred_bucket_col, type_col]).copy()
    if work.empty:
        raise ValueError('no rows with valid bucket ids')
    work[true_bucket_col] = work[true_bucket_col].astype(int)
    work[pred_bucket_col] = work[pred_bucket_col].astype(int)
    work[type_col] = pd.to_numeric(work[type_col], errors='coerce').astype(int)
    num_buckets = len(BUCKETS)
    bucket_names = [bucket_label(i) for i in range(num_buckets)]
    if type_labels is None:
        type_labels = {0: 'Conversation (type 0)', 1: 'Code (type 1)', 2: 'Summarization (type 2)'}
    panel_specs = [(work[type_col] == 0, type_labels.get(0, 'type 0')), (work[type_col] == 1, type_labels.get(1, 'type 1')), (work[type_col] == 2, type_labels.get(2, 'type 2')), (pd.Series(True, index=work.index), 'All')]
    w, h = _figsize(350, height=height)
    fig, axes = plt.subplots(1, 4, figsize=(w * 4.2, h * 1.15))
    axes = np.atleast_1d(axes)
    subtitle = 'buckets from classifier columns' if has_classifier_buckets else 'pred bucket derived from predicted tokens (re-run hybrid predict for classifier buckets)'
    for ax, (mask, panel_title) in zip(axes, panel_specs):
        sub = work.loc[mask]
        true_ids = sub[true_bucket_col].to_numpy(dtype=int)
        pred_ids = sub[pred_bucket_col].to_numpy(dtype=int)
        metrics = _bucket_classification_metrics(true_ids, pred_ids, num_buckets)
        matrix = _bucket_confusion_matrix(true_ids, pred_ids, num_buckets)
        display = matrix.astype(float)
        if normalize and matrix.sum() > 0:
            row_sums = matrix.sum(axis=1, keepdims=True)
            with np.errstate(divide='ignore', invalid='ignore'):
                display = np.divide(matrix, row_sums, where=row_sums > 0)
        if sub.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        else:
            sns.heatmap(display, ax=ax, cmap=style_cmap, vmin=0.0, vmax=1.0 if normalize else None, cbar=ax is axes[-1], xticklabels=bucket_names, yticklabels=bucket_names, square=True, linewidths=0.5, linecolor='white')
            ax.set_xlabel('Predicted bucket')
            ax.set_ylabel('True bucket')
            stats_text = 'n=%d\nacc=%.1f%%\nadj=%.1f%%' % (metrics['n'], 100.0 * metrics['accuracy'], 100.0 * metrics['adjacent_accuracy'])
            ax.text(0.02, 1.02, stats_text, transform=ax.transAxes, ha='left', va='bottom', fontsize=9, bbox=dict(facecolor='white', edgecolor='black', alpha=0.85, boxstyle='round,pad=0.3'))
        ax.set_title(panel_title)
    suptitle = title or 'Hybrid bucket classification'
    fig.suptitle('%s\n(%s)' % (suptitle, subtitle), y=1.08)
    plt.tight_layout()
    if show:
        plt.show()
    return (fig, axes)

def plot_residuals(y_true, y_pred, title, height=None):
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=_figsize(350, height=height))
    ax.scatter(y_pred, residuals, alpha=0.5, color=style_cmap(0.95), s=3)
    ax.axhline(0, color=style_cmap(0), linestyle='--')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Residual')
    ax.set_title(title)
    plt.tight_layout()

def plot_nn_learning_curve(train_out_or_history, title='Seq NN learning curve', show=True, height=None):
    if hasattr(train_out_or_history, 'history'):
        h = train_out_or_history.history
        best_epoch = None
        best_val = None
    elif isinstance(train_out_or_history, dict) and 'fit_history' in train_out_or_history:
        h = train_out_or_history['fit_history']
        best_epoch = train_out_or_history.get('best_epoch')
        best_val = train_out_or_history.get('best_val_loss')
    else:
        h = train_out_or_history
        best_epoch = None
        best_val = None
    loss = h.get('loss')
    val_loss = h.get('val_loss')
    if loss is None or val_loss is None:
        raise KeyError("History must contain 'loss' and 'val_loss' (train with validation_split > 0).")
    epochs = np.arange(1, len(loss) + 1)
    if best_epoch is None:
        best_idx = int(np.argmin(val_loss))
        best_epoch = best_idx + 1
    if best_val is None:
        best_val = float(val_loss[int(best_epoch) - 1])
    fig, ax = plt.subplots(figsize=_figsize(350, height=height))
    ax.plot(epochs, loss, label='Train loss', color=style_cmap(0.2), linewidth=1.5)
    ax.plot(epochs, val_loss, label='Validation loss', color=style_cmap(0.75), linewidth=1.5)
    ax.axvline(best_epoch, color=style_cmap(0), linestyle='--', linewidth=1.2, label=f'Best val (epoch {best_epoch}, loss={best_val:.6g})')
    ax.scatter([best_epoch], [best_val], color=style_cmap(0), s=36, zorder=5, marker='o', edgecolors='white', linewidths=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title(title)
    _legend(ax, loc='best', fontsize=8)
    plt.tight_layout()
    if show:
        plt.show()
    return (fig, ax)

def plot_training_association_comparison(runs, title='', show=True, height=None):
    if not runs:
        raise ValueError('runs must be a non-empty list')
    rows = []
    labels = []
    for item in runs:
        label = item['label']
        result = item['result']
        series = item.get('series', 'auto')
        y_true, y_pred = y_true_y_pred_from_train_result(result, series=series)
        m = prediction_association_metrics(y_true, y_pred)
        rows.append({'label': label, **m})
        labels.append(label)
    w, h = _figsize(350, height=height)
    fig, axes = plt.subplots(1, 4, figsize=(w * 4, h * 1.1))
    axes = np.atleast_1d(axes)
    metric_specs = [('pearson_r', 'Pearson r', (-1.05, 1.05)), ('spearman_rho', 'Spearman ρ', (-1.05, 1.05)), ('kendall_tau', 'Kendall τ', (-1.05, 1.05)), ('underprediction_rate', 'Underprediction rate', (0.0, 1.05))]
    n = len(labels)
    x = np.arange(n, dtype=float)
    width = min(0.7, 0.9 / max(n, 1))
    colors = [plot_color(i) for i in range(n)]
    for ax, (key, ylab, ylim) in zip(axes, metric_specs):
        vals = [row[key] for row in rows]
        for i, (lab, v, c) in enumerate(zip(labels, vals, colors)):
            if np.isnan(v):
                ax.bar(x[i], 0.0, width=width, color=c, alpha=0.35, edgecolor='black', linewidth=0.5)
                ax.text(x[i], 0.02 * (ylim[1] - ylim[0]), 'n/a', ha='center', va='bottom', fontsize=8)
            else:
                ax.bar(x[i], v, width=width, color=c, edgecolor='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha='right')
        ax.set_ylabel(ylab)
        ax.set_ylim(*ylim)
        ax.axhline(0, color=style_cmap(0), linewidth=0.8, linestyle='-', zorder=0)
        if key == 'underprediction_rate':
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    if title:
        fig.suptitle(title, y=1.02)
    plt.tight_layout()
    if show:
        plt.show()
    return (fig, axes, rows)

def plot_dependence(combined_model, X_test, target_index, target_name, title='', feature_names=None, height=None):
    names = list(feature_names if feature_names is not None else feature_cols)
    n_features = len(names)
    fig, axes = plt.subplots(1, n_features, figsize=(10, 3), sharey=True)
    if n_features == 1:
        axes = np.array([axes])
    power = getattr(combined_model, 'power_model_', None)
    perf = getattr(combined_model, 'perf_model_', None)
    if power is not None and perf is not None:
        est = power if int(target_index) == 0 else perf
        X_pdp = pd.DataFrame(np.asarray(X_test), columns=names)
        PartialDependenceDisplay.from_estimator(est, X_pdp, features=list(range(n_features)), feature_names=names, kind='average', ax=axes, grid_resolution=50)
    else:
        PartialDependenceDisplay.from_estimator(combined_model, X_test, features=list(range(n_features)), feature_names=names, target=target_index, kind='average', ax=axes, grid_resolution=50)
    for index, ax in enumerate(axes):
        if index != 0:
            ax.set_ylabel('')
        lines = ax.get_lines()
        for i, line in enumerate(lines):
            line.set_color(plot_color(i))
    fig.suptitle(title or target_name, y=1.02)
    plt.tight_layout()

def _filter_X_by_feature(X, feature_names, col_identifier, value):
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    names = list(feature_names) if feature_names is not None else list(feature_cols)
    if isinstance(col_identifier, str):
        try:
            col_idx = names.index(col_identifier)
        except ValueError:
            raise ValueError(f'col_identifier {col_identifier!r} not in feature_names {names}')
    else:
        col_idx = int(col_identifier)
    col_vals = X[:, col_idx].astype(float)
    mask = col_vals == float(value)
    return (X[mask], mask)

def _run_pdp(combined_model, X_pdp, target_index, feature_names, n_features, axes, is_dataframe=False):
    names = list(feature_names)
    power = getattr(combined_model, 'power_model_', None)
    perf = getattr(combined_model, 'perf_model_', None)
    if power is not None and perf is not None:
        est = power if int(target_index) == 0 else perf
        if not is_dataframe:
            X_pdp = pd.DataFrame(np.asarray(X_pdp), columns=names)
        PartialDependenceDisplay.from_estimator(est, X_pdp, features=list(range(n_features)), feature_names=names, kind='average', ax=axes, grid_resolution=50)
    else:
        if is_dataframe:
            X_pdp = np.asarray(X_pdp)
        PartialDependenceDisplay.from_estimator(combined_model, X_pdp, features=list(range(n_features)), feature_names=names, target=target_index, kind='average', ax=axes, grid_resolution=50)

def plot_dependence_subset(combined_model, X_test, target_index, target_name, subset_col, subset_value, title='', feature_names=None, height=None):
    names = list(feature_names if feature_names is not None else feature_cols)
    n_features = len(names)
    X_arr = np.asarray(X_test)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(1, -1)
    X_sub, mask = _filter_X_by_feature(X_arr, names, subset_col, subset_value)
    if X_sub.shape[0] == 0:
        fig, axes = plt.subplots(1, 1, figsize=(6, 2))
        axes.text(0.5, 0.5, f'No samples with {subset_col}={subset_value}', ha='center', va='center')
        axes.set_axis_off()
        if title:
            fig.suptitle(title, y=1.02)
        plt.tight_layout()
        return
    fig, axes = plt.subplots(1, n_features, figsize=(10, 3), sharey=True)
    if n_features == 1:
        axes = np.array([axes])
    _run_pdp(combined_model, X_sub, target_index, names, n_features, axes, is_dataframe=False)
    for index, ax in enumerate(axes):
        if index != 0:
            ax.set_ylabel('')
        lines = ax.get_lines()
        for i, line in enumerate(lines):
            line.set_color(plot_color(i))
    fig.suptitle(title or f'{target_name} ({subset_col}={subset_value})', y=1.02)
    plt.tight_layout()

def plot_dependence_decode_heavy(combined_model, X_test, target_index, target_name, title='', feature_names=None, height=None):
    plot_dependence_subset(combined_model, X_test, target_index, target_name, subset_col='input_size', subset_value=1, title=title or f'{target_name} — Decode-heavy (input_size=1)', feature_names=feature_names)

def plot_dependence_prefill_heavy(combined_model, X_test, target_index, target_name, title='', feature_names=None, height=None):
    plot_dependence_subset(combined_model, X_test, target_index, target_name, subset_col='output_size', subset_value=1, title=title or f'{target_name} — Prefill-heavy (output_size=1)', feature_names=feature_names)

def plot_dependence_by_regime(combined_model, X_test, target_index, target_name, feature_names=None, height=None):
    plot_dependence_decode_heavy(combined_model, X_test, target_index, target_name, feature_names=feature_names)
    plot_dependence_prefill_heavy(combined_model, X_test, target_index, target_name, feature_names=feature_names)

def plot_index(y_true, y_pred, title='', y_label='', height=None):
    fig, ax = plt.subplots(figsize=_figsize(350, height=height))
    indices = np.arange(len(y_true))
    ax.plot(indices, y_true, label='True', alpha=0.8, color=style_cmap(0.05))
    ax.plot(indices, y_pred, label='Predicted', alpha=0.8, color=style_cmap(0.95))
    ax.set_xlabel('Sample Index')
    ax.set_ylabel(y_label)
    ax.set_title(title)
    global_legend(ax=ax, ncol=2, title=title)
    plt.tight_layout()
    plt.show()

def plot_residual_hist(y_true, y_pred, title, height=None):
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=_figsize(350, height=height))
    ax.hist(residuals, bins=50, alpha=0.5, edgecolor='black', color=style_cmap(0.95))
    ax.axvline(0, color=style_cmap(0), linestyle='--')
    ax.set_xlim(-8000, 8000)
    ax.set_ylim(0, 40)
    ax.set_xlabel('Residual')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def plot_epp_sweep_tp(df, ilen=18, olen=1000, tps=None, bs=200, save_path=None, title='', height=None):
    if tps is None:
        tps = [1, 2, 4, 8]
    freqs = [880, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800]
    models = df['model'].unique()
    ncols = len(models)
    nrows = 1
    fig, axes = plt.subplots(nrows, ncols, figsize=_figsize(350, height=height))
    if ncols == 1:
        axes = [axes]
    for col, model in enumerate(models):
        ax = axes[col]
        ax.set_title(model)
        for i, tp in enumerate(tps):
            rows = []
            for freq in freqs:
                filtered_df = df[(df['tp'] == str(tp)) & (df['model'] == model) & (df['input_size'] == str(ilen)) & (df['output_size'] == str(olen)) & (df['frequency'] == str(freq)) & (df['batch_size'] == str(bs))]
                if not filtered_df.empty:
                    energy_joules = filtered_df['energy_joules'].iloc[0]
                    rows.append({'freq': freq, 'Y': energy_joules / 200})
            if not rows:
                continue
            x = np.array([r['freq'] for r in rows])
            y = np.array([r['Y'] for r in rows])
            color = plot_color(i)
            if len(x) >= 2:
                spline = interpolate.make_interp_spline(x, y, k=min(3, len(x) - 1))
                xs = np.linspace(x.min(), x.max(), 500)
                ys = spline(xs)
                sns.lineplot(ax=ax, x=xs, y=ys, label=f'{tp}', color=color)
            else:
                sns.lineplot(ax=ax, x=x, y=y, label=f'{tp}', color=color)
        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('Energy Per Prompt (J)')
    if title:
        fig.suptitle(title)
    global_legend(axes[0], title='TP', ncol=len(tps))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, transparent=True)
    plt.show()

def plot_slo_meet(df, tp=1, height=None):
    df['input_output_pair'] = df['input_size'].astype(str) + ' ' + df['output_size'].astype(str)
    df['batch_freq'] = df['batch_size'].astype(str) + '-' + df['frequency'].astype(str)
    freq_values = np.arange(900, 1900, 300)
    batch_sizes = df['batch_size'].unique()
    heatmap_data = []
    for input_output in df['input_output_pair'].unique():
        row_data = []
        for batch_size in batch_sizes:
            for freq in freq_values:
                filtered_df = df[(df['batch_size'] == str(batch_size)) & (df['frequency'] == str(freq)) & (df['input_output_pair'] == input_output)]
                energy_joules = filtered_df['energy_joules'].sum() if not filtered_df.empty else 0
                energy_kj = energy_joules / 1000
                row_data.append(energy_kj)
        heatmap_data.append(row_data)
    heatmap_df = pd.DataFrame(heatmap_data, columns=[f'{batch_size}-{freq}' for batch_size in batch_sizes for freq in freq_values], index=df['input_output_pair'].unique())
    fig, ax = plt.subplots(figsize=_figsize(1100, height=height))
    sns.heatmap(heatmap_df, cmap=style_cmap.reversed(), annot=True, fmt=',.0f', cbar_kws={'label': 'Energy (KJ)'}, ax=ax)
    ax.set_title('Energy KJ Heatmap')
    ax.set_xlabel(f'{CONCURRENT_REQUESTS_LABEL} - Frequency')
    ax.set_ylabel('Input Size - Output Size')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()

def plot_slo_meet(df, tp=1, height=None):
    df['input_output_pair'] = df['input_size'].astype(str) + ' ' + df['output_size'].astype(str)
    df['batch_freq'] = df['batch_size'].astype(str) + '-' + df['frequency'].astype(str)
    freq_values = np.arange(900, 1900, 300)
    batch_sizes = df['batch_size'].unique()
    heatmap_data = []
    for input_output in df['input_output_pair'].unique():
        slo_df = df[(df['input_output_pair'] == input_output) & (df['batch_size'] == str(5)) & (df['frequency'] == str(1800)) & (df['tp'] == str(tp))]
        if slo_df.empty:
            print(f'Warning: No baseline data found for Input-Output Pair {input_output}. Skipping this pair.')
            continue
        slo_tp = slo_df['elapsed_time'].mean()
        row_data = []
        for batch_size in batch_sizes:
            for freq in freq_values:
                filtered_df = df[(df['batch_size'] == str(batch_size)) & (df['frequency'] == str(freq)) & (df['input_output_pair'] == input_output) & (df['tp'] == str(tp))]
                energy_joules = filtered_df['energy_joules'].sum() if not filtered_df.empty else 0
                energy_kj = energy_joules / 1000
                tokens_per_second = filtered_df['elapsed_time'].mean() if not filtered_df.empty else 0
                if tokens_per_second > 0.95 * slo_tp:
                    print(f'Not meeting SLO {slo_tp:.2f}: Input-Output Pair: {input_output}, {CONCURRENT_REQUESTS_LABEL}: {batch_size}, Frequency: {freq}, Tokens per second: {tokens_per_second:.2f}')
                    energy_kj = np.nan
                row_data.append(energy_kj)
        heatmap_data.append(row_data)
    heatmap_df = pd.DataFrame(heatmap_data, columns=[f'{batch_size}-{freq}' for batch_size in batch_sizes for freq in freq_values], index=df['input_output_pair'].unique())
    heatmap_df = heatmap_df.apply(pd.to_numeric, errors='coerce')
    if not all(heatmap_df.dtypes == 'float64'):
        raise ValueError('The heatmap data contains non-numeric values. Please check the data.')
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(heatmap_df, cmap=style_cmap, annot=True, fmt=',.0f', cbar_kws={'label': 'Energy (KJ)'}, ax=ax, mask=heatmap_df.isna())
    ax.set_title('Energy KJ Heatmap')
    ax.set_xlabel(f'{CONCURRENT_REQUESTS_LABEL} - Frequency')
    ax.set_ylabel('Input Size - Output Size')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()

def _ecdf_xy(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (np.array([]), np.array([]))
    v = np.sort(v)
    y = np.arange(1, v.size + 1, dtype=float) / v.size
    return (v, y)

def plot_cdf_ttft(df, ttft_col='ttft_s', *, hue=None, ax=None, title=None, xlabel='TTFT (s)', ylabel='CDF', show=True, height=None):
    if ttft_col not in df.columns:
        raise KeyError(f'Column {ttft_col!r} not in DataFrame. Available: {list(df.columns)}')
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=_figsize(350, height=height))
    else:
        fig = ax.get_figure()
    series = []
    if hue is None:
        x, y = _ecdf_xy(df[ttft_col].to_numpy())
        if x.size:
            series.append((None, x, y))
    else:
        if hue not in df.columns:
            raise KeyError(f'hue column {hue!r} not in DataFrame')
        for key, g in df.groupby(hue, sort=False):
            x, y = _ecdf_xy(g[ttft_col].to_numpy())
            if x.size:
                series.append((key, x, y))
    if not series:
        ax.text(0.5, 0.5, 'no finite TTFT values', ha='center', va='center', transform=ax.transAxes)
    else:
        n = len(series)
        for i, (key, x, y) in enumerate(series):
            color = plot_color(i)
            lab = str(key) if key is not None else 'TTFT'
            ax.step(x, y, where='post', color=color, label=lab)
    ax.set_ylim(0, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or 'TTFT empirical CDF')
    tick_style(ax)
    if series:
        _legend(ax, loc='lower right', frameon=True)
    fig.tight_layout()
    if show:
        plt.show()
    return (fig, ax)

def plot_cdf_tpot(df, tpot_col='tpot_mean_s', *, hue=None, ax=None, title=None, xlabel='TPOT (s / output token)', ylabel='CDF', show=True, height=None):
    if tpot_col not in df.columns:
        raise KeyError(f'Column {tpot_col!r} not in DataFrame. Available: {list(df.columns)}')
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=_figsize(350, height=height))
    else:
        fig = ax.get_figure()
    series = []
    if hue is None:
        x, y = _ecdf_xy(df[tpot_col].to_numpy())
        if x.size:
            series.append((None, x, y))
    else:
        if hue not in df.columns:
            raise KeyError(f'hue column {hue!r} not in DataFrame')
        for key, g in df.groupby(hue, sort=False):
            x, y = _ecdf_xy(g[tpot_col].to_numpy())
            if x.size:
                series.append((key, x, y))
    if not series:
        ax.text(0.5, 0.5, 'no finite TPOT values', ha='center', va='center', transform=ax.transAxes)
    else:
        n = len(series)
        for i, (key, x, y) in enumerate(series):
            color = plot_color(i)
            lab = str(key) if key is not None else 'TPOT'
            ax.step(x, y, where='post', color=color, label=lab)
    ax.set_ylim(0, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or 'TPOT empirical CDF')
    tick_style(ax)
    if series:
        _legend(ax, loc='lower right', frameon=True)
    fig.tight_layout()
    if show:
        plt.show()
    return (fig, ax)

def _aggregate_metric_by_x(df, x_col, value_cols, hue):
    d = df.copy()
    need = [x_col, *value_cols]
    if hue is not None:
        need.append(hue)
    for c in need:
        if c not in d.columns:
            raise KeyError(f'Column {c!r} not in DataFrame (have: {list(d.columns)})')
        if c != hue:
            d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d.dropna(subset=[x_col, *value_cols])
    if d.empty:
        return pd.DataFrame()
    agg_dict = {c: 'mean' for c in value_cols}
    if hue is not None:
        gcols = [hue, x_col]
        out = d.groupby(gcols, as_index=False).agg(agg_dict)
        return out.sort_values(gcols)
    return d.groupby(x_col, as_index=False).agg(agg_dict).sort_values(x_col)

def plot_mean_and_p999_tpot_by_batch_size(df, *, batch_col='batch_size', mean_col='tpot_mean_s', p999_col='tpot_p99_9_s', hue=None, threshold=None, title=None, ylabel='TPOT (s / output token)', xlabel=CONCURRENT_REQUESTS_LABEL, show=True, height=None):
    agg = _aggregate_metric_by_x(df, batch_col, [mean_col, p999_col], hue)
    if agg.empty:
        fig, ax = plt.subplots(figsize=_figsize(350, height=height))
        ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes)
        if show:
            plt.show()
        return (fig, ax)
    fig, ax = plt.subplots(figsize=_figsize(350, height=height))
    if hue is not None:
        keys = list(agg[hue].unique())
        n = len(keys)
        for i, key in enumerate(keys):
            g = agg.loc[agg[hue] == key].sort_values(batch_col)
            color = plot_color(i)
            ax.plot(g[batch_col], g[mean_col], 'o-', color=color, label=f'{key} mean')
            ax.plot(g[batch_col], g[p999_col], 's--', color=color, label=f'{key} 99.9th')
    else:
        g = agg.sort_values(batch_col)
        ax.plot(g[batch_col], g[mean_col], 'o-', color=style_cmap(0.15), label='Mean TPOT')
        ax.plot(g[batch_col], g[p999_col], 's--', color=style_cmap(0.85), label='99.9th pct TPOT')
    if threshold is not None:
        t = float(threshold)
        if np.isfinite(t):
            y_slo = 1.0 + t
            x_min = float(np.nanmin(agg[batch_col].to_numpy(dtype=float)))
            x_max = float(np.nanmax(agg[batch_col].to_numpy(dtype=float)))
            if np.isfinite(x_min) and np.isfinite(x_max):
                ax.plot([x_min, x_max], [y_slo, y_slo], color='red', linestyle=':', linewidth=1.6, marker='o', markersize=5, label='SLO Threshold')
            else:
                ax.axhline(y_slo, color='red', linestyle=':', linewidth=1.6, label='SLO Threshold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or f'TPOT vs {CONCURRENT_REQUESTS_LABEL.lower()} (mean and 99.9th)')
    tick_style(ax)
    _legend(ax, loc='best', frameon=True)
    fig.tight_layout()
    if show:
        plt.show()
    return (fig, ax)

def plot_mean_and_p999_tpot_by_frequency(df, *, freq_col='frequency_mhz', mean_col='tpot_mean_s', p999_col='tpot_p99_9_s', hue=None, threshold=None, title=None, ylabel='TPOT (s / output token)', xlabel='Frequency (MHz)', show=True, height=None):
    agg = _aggregate_metric_by_x(df, freq_col, [mean_col, p999_col], hue)
    if agg.empty:
        fig, ax = plt.subplots(figsize=_figsize(350, height=height))
        ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes)
        if show:
            plt.show()
        return (fig, ax)
    fig, ax = plt.subplots(figsize=_figsize(350, height=height))
    if hue is not None:
        keys = list(agg[hue].unique())
        n = len(keys)
        for i, key in enumerate(keys):
            g = agg.loc[agg[hue] == key].sort_values(freq_col)
            color = plot_color(i)
            ax.plot(g[freq_col], g[mean_col], 'o-', color=color, label=f'{key} mean')
            ax.plot(g[freq_col], g[p999_col], 's--', color=color, label=f'{key} 99.9th')
    else:
        g = agg.sort_values(freq_col)
        ax.plot(g[freq_col], g[mean_col], 'o-', color=style_cmap(0.15), label='Mean TPOT')
        ax.plot(g[freq_col], g[p999_col], 's--', color=style_cmap(0.85), label='99.9th pct TPOT')
    if threshold is not None:
        t = float(threshold)
        if np.isfinite(t):
            y_slo = 1.0 + t
            x_min = float(np.nanmin(agg[freq_col].to_numpy(dtype=float)))
            x_max = float(np.nanmax(agg[freq_col].to_numpy(dtype=float)))
            if np.isfinite(x_min) and np.isfinite(x_max):
                ax.plot([x_min, x_max], [y_slo, y_slo], color='red', linestyle=':', linewidth=1.6, marker='o', markersize=5, label='SLO Threshold')
            else:
                ax.axhline(y_slo, color='red', linestyle=':', linewidth=1.6, label='SLO Threshold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or 'TPOT vs frequency (mean and 99.9th)')
    tick_style(ax)
    ax.tick_params(axis='x', labelrotation=90)
    _legend(ax, loc='best', frameon=True)
    fig.tight_layout()
    if show:
        plt.show()
    return (fig, ax)

def plot_mean_and_p999_ttft_by_batch_size(df, *, batch_col='batch_size', mean_col='ttft_mean_s', p999_col='ttft_p99_9_s', hue=None, threshold=None, title=None, ylabel='TTFT (s)', xlabel=CONCURRENT_REQUESTS_LABEL, show=True, height=None):
    agg = _aggregate_metric_by_x(df, batch_col, [mean_col, p999_col], hue)
    if agg.empty:
        fig, ax = plt.subplots(figsize=_figsize(350, height=height))
        ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes)
        if show:
            plt.show()
        return (fig, ax)
    fig, ax = plt.subplots(figsize=_figsize(350, height=height))
    if hue is not None:
        keys = list(agg[hue].unique())
        n = len(keys)
        for i, key in enumerate(keys):
            g = agg.loc[agg[hue] == key].sort_values(batch_col)
            color = plot_color(i)
            ax.plot(g[batch_col], g[mean_col], 'o-', color=color, label=f'{key} mean')
            ax.plot(g[batch_col], g[p999_col], 's--', color=color, label=f'{key} 99.9th')
    else:
        g = agg.sort_values(batch_col)
        ax.plot(g[batch_col], g[mean_col], 'o-', color=style_cmap(0.15), label='Mean TTFT')
        ax.plot(g[batch_col], g[p999_col], 's--', color=style_cmap(0.85), label='99.9th pct TTFT')
    if threshold is not None:
        t = float(threshold)
        if np.isfinite(t):
            y_slo = 1.0 + t
            x_min = float(np.nanmin(agg[batch_col].to_numpy(dtype=float)))
            x_max = float(np.nanmax(agg[batch_col].to_numpy(dtype=float)))
            if np.isfinite(x_min) and np.isfinite(x_max):
                ax.plot([x_min, x_max], [y_slo, y_slo], color='red', linestyle=':', linewidth=1.6, marker='o', markersize=5, label='SLO Threshold')
            else:
                ax.axhline(y_slo, color='red', linestyle=':', linewidth=1.6, label='SLO Threshold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or f'TTFT vs {CONCURRENT_REQUESTS_LABEL.lower()} (mean and 99.9th)')
    tick_style(ax)
    _legend(ax, loc='best', frameon=True)
    fig.tight_layout()
    if show:
        plt.show()
    return (fig, ax)

def plot_mean_and_p999_ttft_by_frequency(df, *, freq_col='frequency_mhz', mean_col='ttft_mean_s', p999_col='ttft_p99_9_s', hue=None, threshold=None, title=None, ylabel='TTFT (s)', xlabel='Frequency (MHz)', show=True, height=None):
    agg = _aggregate_metric_by_x(df, freq_col, [mean_col, p999_col], hue)
    if agg.empty:
        fig, ax = plt.subplots(figsize=_figsize(350, height=height))
        ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes)
        if show:
            plt.show()
        return (fig, ax)
    fig, ax = plt.subplots(figsize=_figsize(350, height=height))
    if hue is not None:
        keys = list(agg[hue].unique())
        n = len(keys)
        for i, key in enumerate(keys):
            g = agg.loc[agg[hue] == key].sort_values(freq_col)
            color = plot_color(i)
            ax.plot(g[freq_col], g[mean_col], 'o-', color=color, label=f'{key} mean')
            ax.plot(g[freq_col], g[p999_col], 's--', color=color, label=f'{key} 99.9th')
    else:
        g = agg.sort_values(freq_col)
        ax.plot(g[freq_col], g[mean_col], 'o-', color=style_cmap(0.15), label='Mean TTFT')
        ax.plot(g[freq_col], g[p999_col], 's--', color=style_cmap(0.85), label='99.9th pct TTFT')
    if threshold is not None:
        t = float(threshold)
        if np.isfinite(t):
            y_slo = 1.0 + t
            x_min = float(np.nanmin(agg[freq_col].to_numpy(dtype=float)))
            x_max = float(np.nanmax(agg[freq_col].to_numpy(dtype=float)))
            if np.isfinite(x_min) and np.isfinite(x_max):
                ax.plot([x_min, x_max], [y_slo, y_slo], color='red', linestyle=':', linewidth=1.6, marker='o', markersize=5, label='SLO Threshold')
            else:
                ax.axhline(y_slo, color='red', linestyle=':', linewidth=1.6, label='SLO Threshold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or 'TTFT vs frequency (mean and 99.9th)')
    tick_style(ax)
    ax.tick_params(axis='x', labelrotation=90)
    _legend(ax, loc='best', frameon=True)
    fig.tight_layout()
    if show:
        plt.show()
    return (fig, ax)

def plot_energy_joules_by_batch_size(df, *, batch_col='batch_size', energy_col='energy_joules', hue=None, title=None, ylabel='Energy (J)', xlabel=CONCURRENT_REQUESTS_LABEL, show=True, height=None):
    agg = _aggregate_metric_by_x(df, batch_col, [energy_col], hue)
    if agg.empty:
        fig, ax = plt.subplots(figsize=_figsize(350, height=height))
        ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes)
        if show:
            plt.show()
        return (fig, ax)
    fig, ax = plt.subplots(figsize=_figsize(350, height=height))
    if hue is not None:
        keys = list(agg[hue].unique())
        n = len(keys)
        for i, key in enumerate(keys):
            g = agg.loc[agg[hue] == key].sort_values(batch_col)
            color = plot_color(i)
            ax.plot(g[batch_col], g[energy_col], 'o-', color=color, label=str(key))
    else:
        g = agg.sort_values(batch_col)
        ax.plot(g[batch_col], g[energy_col], 'o-', color=style_cmap(0.5), label='Energy (J)')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or f'Energy vs {CONCURRENT_REQUESTS_LABEL.lower()}')
    tick_style(ax)
    _legend(ax, loc='best', frameon=True)
    fig.tight_layout()
    if show:
        plt.show()
    return (fig, ax)

def plot_energy_joules_by_frequency(df, *, freq_col='frequency_mhz', energy_col='energy_joules', hue=None, title=None, ylabel='Energy (J)', xlabel='Frequency (MHz)', show=True, height=None):
    agg = _aggregate_metric_by_x(df, freq_col, [energy_col], hue)
    if agg.empty:
        fig, ax = plt.subplots(figsize=_figsize(350, height=height))
        ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes)
        if show:
            plt.show()
        return (fig, ax)
    fig, ax = plt.subplots(figsize=_figsize(350, height=height))
    if hue is not None:
        keys = list(agg[hue].unique())
        n = len(keys)
        for i, key in enumerate(keys):
            g = agg.loc[agg[hue] == key].sort_values(freq_col)
            color = plot_color(i)
            ax.plot(g[freq_col], g[energy_col], 'o-', color=color, label=str(key))
    else:
        g = agg.sort_values(freq_col)
        ax.plot(g[freq_col], g[energy_col], 'o-', color=style_cmap(0.5), label='Energy (J)')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or 'Energy vs frequency')
    tick_style(ax)
    ax.tick_params(axis='x', labelrotation=90)
    _legend(ax, loc='best', frameon=True)
    fig.tight_layout()
    if show:
        plt.show()
    return (fig, ax)

def plot_output_length_dataset_distributions(df, *, title='', bins=50, show=True, type_labels=None, log_x=False, height=None):
    required = {'ContextTokens', 'GeneratedTokens', 'type'}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f'DataFrame missing columns: {missing}')
    if df.empty:
        raise ValueError('DataFrame is empty')
    if type_labels is None:
        type_labels = {0: 'Conversation (type 0)', 1: 'Code (type 1)', 2: 'Summarization (type 2)'}
    cdf_color = style_cmap(0)
    p50_color = style_cmap(0.25)
    p99_color = style_cmap(0.75)
    type_colors = {0: style_cmap(0.12), 1: style_cmap(0.38), 2: style_cmap(0.62)}
    all_color = style_cmap(0.9)
    rows_spec = [('ContextTokens', 'Input tokens'), ('GeneratedTokens', 'Output tokens')]
    panel_specs = [(df['type'] == 0, type_labels.get(0, 'type 0'), type_colors[0]), (df['type'] == 1, type_labels.get(1, 'type 1'), type_colors[1]), (df['type'] == 2, type_labels.get(2, 'type 2'), type_colors[2]), (pd.Series(True, index=df.index), 'All', all_color)]
    w, h = _figsize(350, height=height)
    fig, axes = plt.subplots(2, 4, figsize=(w * 4, h * 2), sharex='row', sharey='row')
    for row_index, (column, xlabel) in enumerate(rows_spec):
        vals_all = pd.to_numeric(df[column], errors='coerce').dropna().to_numpy(dtype=float)
        bin_edges = np.histogram_bin_edges(vals_all, bins=bins) if vals_all.size else None
        cdf_twins = []
        for col_index, (mask, panel_title, color) in enumerate(panel_specs):
            ax = axes[row_index, col_index]
            sub = pd.to_numeric(df.loc[mask, column], errors='coerce').dropna()
            twin = None
            if sub.empty:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            elif bin_edges is not None:
                ax.hist(sub.to_numpy(dtype=float), bins=bin_edges, alpha=0.5, color=color, edgecolor='black', zorder=1)
                twin = ax.twinx()
                x_sorted = np.sort(sub.to_numpy(dtype=float))
                n = len(x_sorted)
                y_cdf = np.arange(1, n + 1, dtype=float) / n
                twin.fill_between(x_sorted, 0, y_cdf, step='post', alpha=0.2, color=cdf_color, zorder=2, linewidth=0)
                twin.step(x_sorted, y_cdf, where='post', color=cdf_color, linewidth=1.2, zorder=5)
                twin.set_ylim(0, 1)
                twin.tick_params(axis='y', labelright=True)
                p50, p99 = np.percentile(sub.to_numpy(dtype=float), [50, 99])
                ax.axvline(p50, color=p50_color, linestyle='--', linewidth=1.2, zorder=7)
                ax.axvline(p99, color=p99_color, linestyle='--', linewidth=1.2, zorder=7)
            cdf_twins.append(twin)
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Count')
            ax.set_title(panel_title if row_index == 0 else '')
            if log_x:
                ax.set_xscale('log')
        active_twins = [t for t in cdf_twins if t is not None]
        if active_twins:
            base_twin = active_twins[0]
            for t in active_twins[1:]:
                t.sharey(base_twin)
            base_twin.set_ylim(0, 1)
            active_twins[-1].set_ylabel('CDF')
    legend_handles = [Line2D([0], [0], color=p50_color, linestyle='--', linewidth=1.2), Line2D([0], [0], color=p99_color, linestyle='--', linewidth=1.2)]
    style_legend(fig.legend(legend_handles, ['50th percentile', '99th percentile'], loc='upper center', frameon=True, borderaxespad=0.5, ncol=2, bbox_to_anchor=(0.4, 1.06)))
    if title:
        fig.suptitle(title, y=1.02)
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    if show:
        plt.show()
    return (fig, axes)
