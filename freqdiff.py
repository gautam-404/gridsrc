import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker 
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.interpolate import interp1d
import matplotlib as mpl
from ipywidgets import interactive, FloatSlider, fixed
from IPython import display

import os, sys
# sys.path.append('../')
# import param_samples


def load_data(param, param_idx=0, rot="uniform", minisaurus_dir="minisauruses", extra_cols=[]):
    inputs = pd.read_csv(os.path.join(minisaurus_dir, param, f'track_inputs_{param}_{rot}.csv'))
    cols = ['m', 'z', 'v', 'surf_avg_v_rot', 'omega', 'R_eq', 'R_polar', 'omega_c',
            'run_time', 'param',
            'density', 'log_Teff', 'tr_num', 'Myr', 'n1ell0m0', 'n2ell0m0', 'n3ell0m0', 'n4ell0m0', 'n5ell0m0',
            'n6ell0m0', 'n7ell0m0', 'n8ell0m0', 'n9ell0m0', 'n10ell0m0', 'n1ell1m0',
            'n2ell1m0', 'n3ell1m0', 'n4ell1m0', 'n5ell1m0', 'n6ell1m0', 'n7ell1m0',
            'n8ell1m0', 'n9ell1m0', 'n10ell1m0', 'n1ell2m0', 'n2ell2m0', 'n3ell2m0',
                'n4ell2m0', 'n5ell2m0', 'n6ell2m0', 'n7ell2m0', 'n8ell2m0', 'n9ell2m0',
                'n10ell2m0', 'n1ell3m0', 'n2ell3m0', 'n3ell3m0', 'n4ell3m0', 'n5ell3m0',
                    'n6ell3m0', 'n7ell3m0', 'n8ell3m0', 'n9ell3m0', 'n10ell3m0',
            'n1ell1dfreq_rot', 'n2ell1dfreq_rot', 'n3ell1dfreq_rot', 'n4ell1dfreq_rot', 'n5ell1dfreq_rot',
            'n6ell1dfreq_rot', 'n7ell1dfreq_rot', 'n8ell1dfreq_rot', 'n9ell1dfreq_rot',
            'n10ell1dfreq_rot', 'n1ell2dfreq_rot', 'n2ell2dfreq_rot', 'n3ell2dfreq_rot',
                'n4ell2dfreq_rot', 'n5ell2dfreq_rot', 'n6ell2dfreq_rot', 'n7ell2dfreq_rot',
                    'n8ell2dfreq_rot', 'n9ell2dfreq_rot', 'n10ell2dfreq_rot', 'n1ell3dfreq_rot',
                'n2ell3dfreq_rot', 'n3ell3dfreq_rot', 'n4ell3dfreq_rot', 'n5ell3dfreq_rot',
                    'n6ell3dfreq_rot', 'n7ell3dfreq_rot', 'n8ell3dfreq_rot', 'n9ell3dfreq_rot',
                    'n10ell3dfreq_rot', 'Dnu', 'eps']
    if len(extra_cols) > 0:
        cols += extra_cols
    previously_saved = False
    if os.path.isfile(os.path.join(minisaurus_dir, param, f'minisaurus_{param}_{rot}.feather')):
        previously_saved = True
        df_t = pd.read_feather(os.path.join(minisaurus_dir, param, f'minisaurus_{param}_{rot}.feather'))
        df_all = df_t
        for col in cols:
            if col not in df_t.columns:
                previously_saved = False
                break
    if previously_saved == False:
        df_all = pd.read_csv(os.path.join(minisaurus_dir, param, f'minisaurus_{param}_{rot}.csv'))
        l_max = 3
        for n in range(1, 11):
            for l in range(0, l_max+1):
                for m in range(-l, l+1):
                    if f"n{n}ell{l}m0" in df_all.columns.values:
                        # df_all[f"n{n}ell{l}m{m}"] = df_all[f"n{n}ell{l}m0"] + m*df_all[f"n{n}ell{l}dfreq_rot"]
                        df_all = df_all.assign(**{f"n{n}ell{l}m{m}": df_all[f"n{n}ell{l}m0"] + m*df_all[f"n{n}ell{l}dfreq_rot"]})

        try:
            df_t = df_all[cols].copy()
        except KeyError:
            # df_all['run_time'] = 0
            df_all = df_all.assign(run_time=np.repeat(0, len(df_all)))
            df_t = df_all[cols].copy()

        # df_t['param_value'] = df_t.param.apply(lambda x: list(eval(x).values())[param_idx])
        df_t = df_t.assign(param_value=df_t.param.apply(lambda x: list(eval(x).values())[param_idx]))
        df_t.to_feather(os.path.join(minisaurus_dir, param, f'minisaurus_{param}_{rot}.feather'))

    cols0 = [col for col in df_t.columns.values if "ell0m" in col and 'dfreq_rot' not in col]
    cols1 = [col for col in df_t.columns.values if "ell1m" in col and 'dfreq_rot' not in col]
    cols2 = [col for col in df_t.columns.values if "ell2m" in col and 'dfreq_rot' not in col]
    cols3 = [col for col in df_t.columns.values if "ell3m" in col and 'dfreq_rot' not in col]
    return df_t, cols0, cols1, cols2, cols3, inputs, df_all


def fractional_f(ages, params, df_t, cols0, cols1, cols2, cols3, ref=0):
    df_list = []
    for (M, V, Z), group in df_t.groupby(['m', 'v', 'z']):
        for param in group.param_value.unique():
            this_df = group[group.param_value == param].sort_values(by=['Myr']).reset_index()
            new_df = pd.DataFrame({'Myr': ages})
            for col in [col for col in group.columns if col not in ['index', 'Myr', 'm', 'v', 'z', 'param', 'param_value', 'tr_num']]:
                f = interp1d(this_df["Myr"], this_df[col], kind='slinear', fill_value='extrapolate')
                new_df[col] = f(ages)
            new_df['param_value'] = param
            new_df['M'] = M
            new_df['V'] = V
            new_df['Z'] = Z
            new_df['run_time'] = this_df.run_time.values[0]
            df_list.append(new_df)

    df_master = pd.concat(df_list, ignore_index=True)
    cols = cols0 + cols1 + cols2 + cols3
    new_cols_df = pd.DataFrame(index=df_master.index)
    for col in ['mean_ff'] + [f'{col}_ff' for col in cols]:
        new_cols_df[col] = np.nan
    df_master = pd.concat([df_master, new_cols_df], axis=1)

    if isinstance(ref, int):
        df_refs = [df_master[df_master.param_value == params[ref]]]
    if isinstance(ref, list):
        df_refs = [df_master[df_master.param_value == params[r]] for r in ref]
    for (M, V, Z, param), group in df_master.groupby(['M', 'V', 'Z', 'param_value']):
        for df_ref in df_refs:
            df_alt = group
            freqs_ref = df_ref[(df_ref['M'] == M) & (df_ref['V'] == V) & (df_ref['Z'] == Z)][cols].values
            freqs_alt = df_alt[cols].values
            diff = np.nan_to_num((freqs_alt - freqs_ref) / freqs_ref, nan=0)
            mean_diff = np.mean(diff, axis=1)

            for i, col in enumerate(cols):
                df_master.loc[group.index, f'{col}_ff'] = diff[:, i]
            df_master.loc[group.index, 'mean_ff'] = mean_diff
    return df_master


def get_filtered_columns(df, patterns):
    """
    Filter columns in dataframe based on multiple patterns.
    """
    cols = []
    for pattern in patterns:
        cols += [col for col in df.columns if pattern in col and '_ff' not in col]
    return cols

def configure_plot_style(use_linestyles, params, transparent, bgcolor):
    """
    Configure plot style based on parameters.
    """
    if use_linestyles:
        linestyles = [(0, (3, 1, 1, 1, 1, 1)), 'solid', '--', '-.', ':']
        palette = sns.color_palette("colorblind", len(params))
    else:
        linestyles = ['solid'] * len(params)
        palette = sns.color_palette("magma_r", len(params))
    
    if transparent:
        color = bgcolor
        plt.rcParams.update({'text.color': color, 'axes.labelcolor': color,
                             'xtick.color': color, 'ytick.color': color,
                             'axes.facecolor': bgcolor})
    else:
        plt.rcParams.update(plt.rcParamsDefault)
    
    return palette, linestyles

def add_colorbar(fig, axes, params, param_str, ref, palette, use_linestyles=False):
    if use_linestyles:
        if isinstance(axes) == np.ndarray:
            handles, labels = axes.flatten()[0].get_legend_handles_labels()
        elif isinstance(axes, list):
            handles, labels = axes[0].get_legend_handles_labels()
        else:
            handles, labels = axes.get_legend_handles_labels()
        unique = [(handle, label) for i, (handle, label) in enumerate(zip(handles, labels)) if label not in labels[:i]]
        fig.legend(*zip(*unique), loc='center right', bbox_to_anchor=(1.22, 0.5), title=param_str, prop={'size':20}, title_fontsize=20)
        return fig, axes, None
    else:
        if isinstance(params[0], float) or isinstance(params[0], int):
            Z = [[0,0],[0,0]]
            levels = np.array(sorted(params)+[params[-1]+np.diff(params)[0]])
            contour = plt.contourf(Z, levels, cmap=mpl.colors.ListedColormap(palette))
            level_step = np.diff(levels)[0]/2
            cb = plt.colorbar(contour, ticks=levels+level_step, ax=axes)
            cb.set_label(rf'{param_str}', fontsize=25)
            cb.set_ticklabels([f"{level:.3f}" for level in levels], fontsize=16)
            if isinstance(ref, int):
                cb.ax.hlines(params[ref]+level_step, 0, 1, color='white', linestyle='--', linewidth=2, alpha=0.8)
            elif isinstance(ref, list):
                for r in ref:
                    cb.ax.hlines(params[r]+level_step, 0, 1, color='white', linestyle='--', linewidth=2, alpha=0.8)
            cb.ax.hlines(params[ref]+level_step, 0, 1, color='white', linestyle='--', linewidth=2, alpha=0.8)
        elif isinstance(params[0], str):
            cb = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=1), cmap=mpl.colors.ListedColormap(palette)), 
                                ax=axes, ticks=np.linspace(0, 1, len(params)+1)[:-1]+0.5/(len(params)+1))
            cb.set_label(rf'{param_str}', fontsize=25)
            cb.set_ticklabels(params, fontsize=16)
            if isinstance(ref, int):
                cb.ax.hlines((np.linspace(0, 1, len(params)+1)[:-1]+0.5/(len(params)+1))[ref], 0, 1, color='white', linestyle='--', linewidth=2, alpha=0.8)
            elif isinstance(ref, list):
                for r in ref:
                    cb.ax.hlines((np.linspace(0, 1, len(params)+1)[:-1]+0.5/(len(params)+1))[r], 0, 1, color='white', linestyle='--', linewidth=2, alpha=0.8)
            cb.ax.hlines((np.linspace(0, 1, len(params)+1)[:-1]+0.5/(len(params)+1))[ref], 0, 1, color='white', linestyle='--', linewidth=2, alpha=0.8)
        return fig, axes, cb

def plot_fdf(fig, ax, df_master, m, z, v, age, params, param_name='param',
                                param_str='param', ref=0, use_linestyles=False, transparent=False, bgcolor='white', colorbar=True,
                                xlim=(5, 115), ylim=(-5, 5), swap_axes=False):
    """
    Plots the frequency differences for given parameters.
    """
    patterns = ['ell0m', 'ell1m', 'ell2m', 'ell3m']
    cols = get_filtered_columns(df_master, patterns)

    palette, linestyles = configure_plot_style(use_linestyles, params, transparent, bgcolor)
    markers = ['o', '^', 's', 'd']
    if swap_axes:
        update_fdf_y(ax, df_master, age, m, z, v, params, ref, palette, markers, xlim, ylim)
    else:
        update_fdf_x(ax, df_master, age, m, z, v, params, ref, palette, markers, xlim, ylim)
    if colorbar:
        fig, ax, cb = add_colorbar(fig, ax, params, param_str, ref, palette)
    else:
        cb = None
    return fig, ax, cb


def animate_fdf(fig, ax, df_master, m, z, v, ages, params, 
                                   param_name='param', param_str='param', ref=0, 
                                   use_linestyles=False, transparent=False, bgcolor='white', 
                                   xlim=(5, 115), ylim=(-5, 5), skipstep=10):
    """
    Creates an animation showing the frequency differences for different ages.
    """
    palette, linestyles = configure_plot_style(use_linestyles, params, transparent, bgcolor)
    markers = ['o', '^', 's', 'd'] # Define markers outside the function if they are shared with plot_fdf

    fig, ax, cb = add_colorbar(fig, ax, params, param_str, ref, palette)

    def update(age):
        return update_fdf_y(ax, df_master, age, m, z, v, params, ref, palette, markers, xlim, ylim)
    
    d_age = ages[1] - ages[0] 
    ages = np.arange(ages[0], ages[-1]+d_age, d_age*skipstep)
    ani = mpl.animation.FuncAnimation(fig, update, frames=ages, repeat=True)
    return ani


def update_fdf_y(ax, df_master, age, m, z, v, params, ref, palette, markers, xlim, ylim):
    """
    Updates the plot for a specific age.
    """
    ax.clear()
    age = np.round(age, decimals=4)  # Ensure age is rounded for comparison
    df = df_master.query(f"M=={m} and V=={v} and Z=={z} and Myr=={age}")
    df_ref = df.query(f"param_value=={params[ref]}")
    cols = get_filtered_columns(df_master, ['ell0m', 'ell1m', 'ell2m', 'ell3m'])

    l_max = 3
    df = df_master.query(f"M=={m} and V=={v} and Z=={z} and Myr=={age}")
    df_ref = df.query(f"param_value=={params[ref]}")
    freqs = df_ref[cols]
    for i, param in enumerate(params):
        df_this = df.query(f"param_value=={param}")
        for n in range(1, 11):
            for l in range(l_max+1):
                k = l+n
                col = cols[k]
                ax.scatter(freqs[col], 100*df_this[f'{col}_ff'],  marker=markers[l], s=100, color=palette[i], label=l, alpha=0.8)

    handles, labels = ax.get_legend_handles_labels()
    unique = [(handle, label) for i, (handle, label) in enumerate(zip(handles, labels)) if label not in labels[:i]]
    ax.legend(*zip(*unique), loc='upper right', title=r'$\ell$', prop={'size':15}, title_fontsize=20, ncol=2)

    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.set_title(f'Age: {age} Myrs', fontsize=25, weight='bold')
    ax.set_xlabel(r"$\bf{f, \ \ \rm{d}^{-1}}$", size=25, weight='bold')
    ax.set_ylabel(r'$\bf{\delta f/f}$ (%)', size=25, weight='bold')
    ax.axhline(0, ls='dashed', color='grey')
    ax.tick_params(axis='both', which='major', labelsize=20)
    return ax

def update_fdf_x(ax, df_master, age, m, z, v, params, ref, palette, markers, xlim, ylim):
    """
    Updates the plot for a specific age.
    """
    ax.clear()
    age = np.round(age, decimals=4)  # Ensure age is rounded for comparison
    df = df_master.query(f"M=={m} and V=={v} and Z=={z} and Myr=={age}")
    df_ref = df.query(f"param_value=={params[ref]}")
    cols = get_filtered_columns(df_master, ['ell0m', 'ell1m', 'ell2m', 'ell3m'])

    l_max = 3
    df = df_master.query(f"M=={m} and V=={v} and Z=={z} and Myr=={age}")
    df_ref = df.query(f"param_value=={params[ref]}")
    freqs = df_ref[cols]
    for i, param in enumerate(params):
        df_this = df.query(f"param_value=={param}")
        for n in range(1, 11):
            for l in range(l_max+1):
                k = l+n
                col = cols[k]
                ax.scatter(100*df_this[f'{col}_ff'], freqs[col],  marker=markers[l], s=100, color=palette[i], label=l, alpha=0.8)

    handles, labels = ax.get_legend_handles_labels()
    unique = [(handle, label) for i, (handle, label) in enumerate(zip(handles, labels)) if label not in labels[:i]]
    ax.legend(*zip(*unique), loc='upper right', title=r'$\ell$', prop={'size':15}, title_fontsize=20, ncol=2)

    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.set_title(f'Age: {age} Myrs', fontsize=25, weight='bold')
    ax.set_ylabel(r"$\bf{f, \ \ \rm{d}^{-1}}$", size=25, weight='bold')
    ax.set_xlabel(r'$\bf{\delta f/f}$ (%)', size=25, weight='bold')
    ax.axvline(0, ls='dashed', color='grey')
    ax.tick_params(axis='both', which='major', labelsize=20)
    return ax

def update_fdf_y(ax, df_master, age, m, z, v, params, ref, palette, markers, xlim, ylim):
    """
    Updates the plot for a specific age.
    """
    ax.clear()
    age = np.round(age, decimals=4)  # Ensure age is rounded for comparison
    df = df_master.query(f"M=={m} and V=={v} and Z=={z} and Myr=={age}")
    df_ref = df.query(f"param_value=={params[ref]}")
    cols = get_filtered_columns(df_master, ['ell0m', 'ell1m', 'ell2m', 'ell3m'])

    l_max = 3
    df = df_master.query(f"M=={m} and V=={v} and Z=={z} and Myr=={age}")
    df_ref = df.query(f"param_value=={params[ref]}")
    freqs = df_ref[cols]
    for i, param in enumerate(params):
        df_this = df.query(f"param_value=={param}")
        for n in range(1, 11):
            for l in range(l_max+1):
                k = l+n
                col = cols[k]
                ax.scatter(freqs[col], 100*df_this[f'{col}_ff'],  marker=markers[l], s=100, color=palette[i], label=l, alpha=0.8)

    handles, labels = ax.get_legend_handles_labels()
    unique = [(handle, label) for i, (handle, label) in enumerate(zip(handles, labels)) if label not in labels[:i]]
    ax.legend(*zip(*unique), loc='upper right', title=r'$\ell$', prop={'size':15}, title_fontsize=20, ncol=2)

    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.set_title(f'Age: {age} Myrs', fontsize=25, weight='bold')
    ax.set_xlabel(r"$\bf{f, \ \ \rm{d}^{-1}}$", size=25, weight='bold')
    ax.set_ylabel(r'$\bf{\delta f/f}$ (%)', size=25, weight='bold')
    ax.axhline(0, ls='dashed', color='grey')
    ax.tick_params(axis='both', which='major', labelsize=20)
    return ax


def interactive_fdf(df_master, m, z, v, ages, params, param_name, param_str, ref, use_linestyles=False, transparent=False, bgcolor='white', xlim=(5, 115), ylim=(-5, 5), figsize=(14, 7), **widget_kwargs):
    age_min, age_max = min(ages), max(ages)
    d_age = ages[1] - ages[0]

    palette, linestyles = configure_plot_style(use_linestyles, params, transparent, bgcolor)
    markers = ['o', '^', 's', 'd']

    def plot_with_age(age, figsize):
        fig, ax = plt.subplots(figsize=figsize)
        update_fdf_y(ax, df_master, age, m, z, v, params, ref, palette, markers, xlim, ylim)
        fig, ax, cb = add_colorbar(fig, ax, params, param_str, ref, palette)
        plt.show()

    age_slider = FloatSlider(value=age_min, min=age_min, max=age_max, step=d_age, description='Age:', **widget_kwargs)
    
    # Use interactive to create a widget for the age slider
    interactive_plot = interactive(plot_with_age, age=age_slider, figsize=fixed(figsize))
    return interactive_plot


def plot_mean_ff(fig, ax, df_master, m, z, v, age, ages, params, param_name='param', param_str='param', ref=0, use_linestyles=False, transparent=False, bgcolor='white'):
    cols0 = [f for f in df_master.columns.values if ("ell0m" in f) and ('_ff' not in f)]
    cols1 = [f for f in df_master.columns.values if ("ell1m" in f) and ('_ff' not in f)]
    cols2 = [f for f in df_master.columns.values if ("ell2m" in f) and ('_ff' not in f)]
    cols3 = [f for f in df_master.columns.values if ("ell3m" in f) and ('_ff' not in f)]

    if use_linestyles:
        linestyle_tuple = [(0, (3, 1, 1, 1, 1, 1)), 'solid', '--', '-.', ':']
        palette = sns.color_palette("colorblind", len(params))
    else:
        palette = sns.color_palette("magma_r", len(params))
        linestyle_tuple = ['solid'] * len(params)

    if transparent:
        COLOR = bgcolor
        rc1 = {'text.color': COLOR, 'axes.labelcolor': COLOR, 'xtick.color': COLOR, 'ytick.color': COLOR}
        plt.rcParams.update(rc1)
        ax.set_facecolor(bgcolor)
        ax.grid(True, color=COLOR, linestyle='--', linewidth=1, alpha=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor(COLOR)
    else:
        mpl.rcParams.update(mpl.rcParamsDefault)
    
    l_max = 3
    min_val, max_val = 0, 0
    df = df_master.query(f"M=={m} and V=={v} and Z=={z}")
    df_ref = df.query(f"param_value=={params[ref]}")
    cols = cols0+cols1+cols2+cols3
    freqs = df_ref[cols]
    for i, param in enumerate(params):
        df_this = df.query(f"param_value=={param}")
        ax.plot(ages, 100*df_this['mean_ff'], label=param, color=palette[i])
        min_val = min(min_val, 100*df_this['mean_ff'].min())
        max_val = max(max_val, 100*df_this['mean_ff'].max())
    df_this = df.query(f"param_value=={0.014}")
    ax.scatter(age, 100*df_this.query(f'Myr=={age}')['mean_ff'], s=30, color='black', zorder=3)

    # ax.set_ylim(min_val-0.1, max_val+0.1)
    lim = max(abs(min_val), abs(max_val))
    ax.set_ylim(-1.1*lim, 1.1*lim)
    ax.set_xlim(5, 25)
    
    ax.axhline(0, ls='dashed', color='grey')
    ax.set_xlabel('Age (Myr)', size=25, weight='bold')
    ax.set_ylabel(r'$\bf{\delta f/f}$ (%)', size=25, weight='bold')
    ax.tick_params(axis='both', which='major', labelsize=20)

    fig, ax, cb = add_colorbar(fig=fig, axes=ax, params=params, param_str=param_str, ref=ref, palette=palette, use_linestyles=use_linestyles)
    return fig, ax, cb


def plot_meanff_all(df_master, ages, params, y_major_base=2, y_minor_base=None, x_major_base=5, x_minor_base=None, param_name='param', param_str='param', ref=0, use_linestyles=False):
    row, col, slice_ = 'M', 'V', 'Z'
    M_sample, V_sample, Z_sample = sorted(df_master[row].unique()), sorted(df_master[col].unique()), sorted(df_master[slice_].unique())
    rows, cols, slices = len(M_sample), len(V_sample), len(Z_sample)

    fig = plt.figure(figsize=(22, 16))

    if use_linestyles:
        linestyle_tuple = [(0, (3, 1, 1, 1, 1, 1)), 'solid', '--', '-.', ':']
        palette = sns.color_palette("colorblind", len(params))
    else:
        palette = sns.color_palette("magma_r", len(params))
        linestyle_tuple = ['solid']*len(params)
    axes = np.zeros((slices, rows, cols), dtype=object)
    outer = gridspec.GridSpec(rows, cols, wspace=0.1, hspace=0.1)
    for i in range(rows):
        max_values = []
        min_values = []
        for j in range(cols):
            inner = gridspec.GridSpecFromSubplotSpec(slices, 1, subplot_spec=outer[i * cols + j], wspace=0, hspace=0)
            for k in range(slices):
                ax = plt.Subplot(fig, inner[k])
                ax.grid(axis='both', which='both', visible=True, alpha=0.3)
                ax.set_xlim(ages[0], ages[-1])
                if k != slices-1:
                    ax.set_xticklabels([])
                    ax.xaxis.set_ticks_position('none')
                    if k == 0:
                        ax.spines[['bottom']].set_visible(False)
                    elif k != slices-1:
                        ax.spines[['top', 'bottom']].set(alpha=0.5)
                else:
                    if i != rows-1:
                        ax.set_xticklabels([])
                        ax.xaxis.set_ticks_position('none')
                    ax.tick_params(axis='x', labelsize=16)
                    ax.spines[['top']].set(alpha=0.5)
                if j == cols-1:
                    if k == int(slices/2):
                        ax.set_ylabel(fr"{row} = {M_sample[i]:.2f} $M_\odot$", fontsize=20)
                        ax.yaxis.set_label_position("right")
                    ax.text(0.72, 0.75, f"{slice_} = {Z_sample[k]:.3f}", transform=ax.transAxes, fontsize=20)
                if k == 0 and i == 0:
                    ax.set_title(f"{col} = {V_sample[j]} km/s", fontsize=20)
                if j != 0:
                    ax.set_yticklabels([])
                    ax.yaxis.set_ticks_position('none')
                ax.tick_params(axis='y', labelsize=16)
                fig.add_subplot(ax)
                axes[k][i][j] = ax
                
                for idx in range(len(params)):
                    mean_ff = df_master[(df_master[row] == M_sample[i]) & (df_master[col] == V_sample[j]) & (df_master[slice_] == Z_sample[k]) & (df_master['param_value'] == params[idx])].sort_values(by='Myr')['mean_ff'].values
                    if len(mean_ff) > 0:
                        ax.plot(ages, 100*np.array(mean_ff), color=palette[idx], linestyle=linestyle_tuple[idx], label=params[idx], lw=2, alpha=0.8)
                        max_values.append(max(100*np.array(mean_ff)))
                        min_values.append(min(100*np.array(mean_ff)))

        for ax in axes[:, i, :].flatten():
            if len(mean_ff) > 0:
                min_ = min(min_values)
                max_ = max(max_values)
                ax.set_ylim(min_-0.1, max_+0.1)
            ax.yaxis.set_major_locator(plticker.MultipleLocator(base=y_major_base))
            if y_minor_base is not None:
                ax.yaxis.set_minor_locator(plticker.MultipleLocator(base=y_minor_base))
            ax.xaxis.set_major_locator(plticker.MultipleLocator(base=x_major_base))
            if x_minor_base is not None:
                ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=x_minor_base))             
    fig.supxlabel(r'$\bf{Age \ (Myr)}$', fontsize=22)
    fig.supylabel(r'$\bf{\langle{\delta f/f} \rangle} \ \rm{(\%)}$', fontsize=22)
    fig.suptitle(param_name, fontsize=25)
    outer.tight_layout(fig, rect=[0.01, 0.01, 1, 1])

    cb = add_colorbar(fig=fig, axes=axes, params=params, param_str=param_str, ref=ref, palette=palette, use_linestyles=use_linestyles)
    return fig, axes, None


#### New functions for plotting fdf
def update_comp_plots(age, axes, df_master, ages, m, z, v, params, param_name, param_str, ref, interactive=False, fig=None, colors=None, subplot_params=['pp', 'cno'], 
           subplot_labels=[r"L$_{pp}$ (L$_{\odot}$)", r"L$_{CNO}$ (L$_{\odot}$)'])"], ylim_fdf=(10, 100), xlim_fdf=(-3, 3)):
    """
    Function to update the fractional differences plot for a given age. 
    """
    if colors is None:
        sns.color_palette("magma_r", len(params))
    [ax.cla() for ax in axes.flatten()]
    markers = ['o', '^', 's', 'd']
    ylims_max = []
    ylims_min = []
    for i, subplot_param in enumerate(subplot_params):
        if 'log' in subplot_param or 'cno' in subplot_param or 'pp' in subplot_param:
            ylims_max.append([10**df_master.query(f"M=={m_i} and Z=={z_i} and V=={v_i}")[subplot_param].max() for m_i, z_i, v_i in zip(m, z, v)])
            ylims_min.append([10**df_master.query(f"M=={m_i} and Z=={z_i} and V=={v_i}")[subplot_param].min() for m_i, z_i, v_i in zip(m, z, v)])
        else:
            ylims_max.append([df_master.query(f"M=={m_i} and Z=={z_i} and V=={v_i}")[subplot_param].max() for m_i, z_i, v_i in zip(m, z, v)])
            ylims_min.append([df_master.query(f"M=={m_i} and Z=={z_i} and V=={v_i}")[subplot_param].min() for m_i, z_i, v_i in zip(m, z, v)])
    for i, ((m, z, v), axes_i) in enumerate(zip(zip(m, z, v), axes)):
        axes_i[0] = update_fdf_x(axes_i[0], df_master, age, m, z, v, params, ref, colors, markers, ylim=ylim_fdf, xlim=xlim_fdf)
        if len(subplot_params) > 0:
            for param_value in params:
                for j, subplot_param in enumerate(subplot_params):
                    ax = axes_i[j+1]
                    if 'log' in subplot_param or 'cno' in subplot_param or 'pp' in subplot_param:
                        ax.plot(ages, 10**df_master.query(f"M=={m} and Z=={z} and V=={v} and param_value=={param_value}")[subplot_param], label=f"{param_str} = {param_value}", color=colors[params.index(param_value)])
                    else:
                        ax.plot(ages, df_master.query(f"M=={m} and Z=={z} and V=={v} and param_value=={param_value}")[subplot_param], label=f"{param_str} = {param_value}", color=colors[params.index(param_value)])
            for j, subplot_param in enumerate(subplot_params):
                ax = axes_i[j+1]
                ax.vlines(age, -1, 100, linestyle="--", color="k")
                ax.set_ylabel(subplot_labels[j], fontsize=20, weight="bold")
                ax.tick_params(axis='both', which='major', labelsize=20)
                ax.set_xlim(5, 40)
                ax.set_ylim(ylims_min[j][i]*0.9, ylims_max[j][i]*1.1)
                if j < len(subplot_params)-1:
                    ax.set_xticklabels([])
        if i > 0:
            for ax in axes_i:
                ax.set_yticklabels([])
                ax.set_ylabel("")
        axes_i[0].set_title(f'M={m}, '+r'Z$_{in}$='+f'{z}, '+r'V$_{in}$='+f'{v}', fontsize=20, weight="bold")
        [ax.grid(alpha=0.3) for ax in axes_i]
    fig._suptitle.set_text(f'Age = {age:.2f} Myr')
    if interactive:
        display.display(fig)
    return axes

def comp_plots(fig, df_master, ages, m, z, v, params, param_name, param_str, ref, age_start_idx=40, subplot_params=[], subplot_labels=[], interactive=True, ylim_fdf=(10, 100), xlim_fdf=(-3, 3)):
    """
    Function to compare the fractional differences for different parameters. Subplots can be added to see the evolution of other history columns.

    Arguments
    ----------
    fig : matplotlib.figure.Figure
        Figure object to plot the fractional differences.
    df_master : pandas.DataFrame
        DataFrame containing the fractional differences for all the parameters.
    ages : numpy.ndarray
        Array containing the ages at which the fractional differences are calculated.
    m : list
        List of masses for which the fractional differences are calculated. 
        Note: Length of the list should be the same as the length of z and v.
    z : list
        List of metallicities for which the fractional differences are calculated. 
        Note: Length of the list should be the same as the length of m and v.
    v : list
        List of overshoot values for which the fractional differences are calculated. 
        Note: Length of the list should be the same as the length of m and z.
    params : list
        List of parameter values for which the fractional differences are calculated.
    ref : int
        Index of the reference model.
    subplot_params : list, optional
        List of history columns to be plotted in the subplots. Default is an empty list.
    subplot_labels : list, optional
        List of labels for the subplots. Default is an empty list.
    interactive : bool, optional
        If True, the plot is interactive. Default is True.
    """
    outer = gridspec.GridSpec(nrows=1, ncols=len(m), width_ratios=[0.8 if i != len(m)-1 else 1 for i in range(len(m))], wspace=0.05, figure=fig)
    if len(subplot_params) > 0:
        cols = [gridspec.GridSpecFromSubplotSpec(nrows=2, ncols=1, subplot_spec=outer[i], hspace=0.3) for i in range(len(m))]
        upper = [gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=cols[i][0], hspace=0.05) for i in range(len(m))]
        lower = [gridspec.GridSpecFromSubplotSpec(nrows=len(subplot_params), ncols=1, subplot_spec=cols[i][1], hspace=0.05) for i in range(len(m))]
        axes = np.array([[fig.add_subplot(upper[i][0])] + [fig.add_subplot(lower[i][j]) for j in range(len(subplot_params))] for i in range(len(m))])
    else:
        axes = np.array([[fig.add_subplot(outer[i])] for i in range(len(m))])
    # plt.close()
    title = fig.suptitle(f'Age = {ages[age_start_idx]:.2f} Myr', fontsize=20, weight="bold", x=0.47, y=0.95)
    fig.align_ylabels()
    colors = sns.color_palette("magma_r", len(params))
    cb = add_colorbar(fig=fig, ax=axes[len(m)-1], params=params, param_str=param_str, palette=colors, ref=ref)

    if interactive:
        from ipywidgets import interactive, fixed
        from ipywidgets import FloatSlider
        return interactive(update_comp_plots, age=FloatSlider(value=ages[age_start_idx], min=ages[age_start_idx], max=ages.max(), step=0.1, description='Age:', layout={'width': '1000px'}), 
                    df_master=fixed(df_master), ages=fixed(ages), axes=fixed(axes), interactive=fixed(True), fig=fixed(fig), colors=fixed(colors), params=fixed(params), param_name=fixed(param_name),
                    param_str=fixed(param_str), ref=fixed(ref), subplot_params=fixed(subplot_params), subplot_labels=fixed(subplot_labels), m=fixed(m), z=fixed(z), v=fixed(v), ylim_fdf=fixed(ylim_fdf), xlim_fdf=fixed(xlim_fdf))
    else:
        from matplotlib.animation import FuncAnimation
        ani = FuncAnimation(fig, update_comp_plots, frames=ages[age_start_idx::10], fargs=[axes, df_master, ages, m, z, v, params, param_name, param_str, ref, interactive, fig, colors, subplot_params, subplot_labels, ylim_fdf, xlim_fdf], repeat=False)
        ani.save(f'../figures/{param_name}_fdf_m{m}_z{z}_v{v}_{subplot_params}.gif', writer='ffmpeg', fps=10)
        return ani
    