import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
import matplotlib as mpl
from ipywidgets import interactive, FloatSlider, fixed
from IPython import display

import os, sys
# sys.path.append('../')
# import param_samples


def load_data(param, param_idx=0, rot="uniform"):
    inputs = pd.read_csv(f'../minisauruses/track_inputs_{param}_{rot}.csv')
    if os.path.isfile(f"../minisauruses/{param}_{rot}.feather"):
        df_t = pd.read_feather(f"../minisauruses/{param}_{rot}.feather")
        df_all = df_t
    else:
        df_all = pd.read_csv(f'../minisauruses/minisaurus_{param}_{rot}.csv')
        cols = ['m', 'z', 'v', 'surf_avg_v_rot', 'surf_avg_omega_div_omega_crit', 
            'run_time', 'param',
            'density', 'log_Teff', 'tr_num', 'Myr', 'n1ell0m0', 'n2ell0m0', 'n3ell0m0', 'n4ell0m0', 'n5ell0m0',
            'n6ell0m0', 'n7ell0m0', 'n8ell0m0', 'n9ell0m0', 'n10ell0m0', 'n1ell1m0',
            'n2ell1m0', 'n3ell1m0', 'n4ell1m0', 'n5ell1m0', 'n6ell1m0', 'n7ell1m0',
            'n8ell1m0', 'n9ell1m0', 'n10ell1m0', 'n1ell2m0', 'n2ell2m0', 'n3ell2m0',
                'n4ell2m0', 'n5ell2m0', 'n6ell2m0', 'n7ell2m0', 'n8ell2m0', 'n9ell2m0',
                'n10ell2m0', 'n1ell3m0', 'n2ell3m0', 'n3ell3m0', 'n4ell3m0', 'n5ell3m0',
                    'n6ell3m0', 'n7ell3m0', 'n8ell3m0', 'n9ell3m0', 'n10ell3m0',
            'n1ell1dfreq', 'n2ell1dfreq', 'n3ell1dfreq', 'n4ell1dfreq', 'n5ell1dfreq',
            'n6ell1dfreq', 'n7ell1dfreq', 'n8ell1dfreq', 'n9ell1dfreq',
            'n10ell1dfreq', 'n1ell2dfreq', 'n2ell2dfreq', 'n3ell2dfreq',
                'n4ell2dfreq', 'n5ell2dfreq', 'n6ell2dfreq', 'n7ell2dfreq',
                    'n8ell2dfreq', 'n9ell2dfreq', 'n10ell2dfreq', 'n1ell3dfreq',
                'n2ell3dfreq', 'n3ell3dfreq', 'n4ell3dfreq', 'n5ell3dfreq',
                    'n6ell3dfreq', 'n7ell3dfreq', 'n8ell3dfreq', 'n9ell3dfreq',
                    'n10ell3dfreq', 'Dnu', 'eps']
        l_max = 3
        for n in range(1, 11):
            for l in range(0, l_max+1):
                for m in range(-l, 0):
                    if f"n{n}ell{l}m0" in df_all.columns.values:
                        df_all = df_all.assign(**{f"n{n}ell{l}mm{abs(m)}": df_all[f"n{n}ell{l}m0"] - df_all[f"n{n}ell{l}dfreq"]})
                        df_all = df_all.assign(**{f"n{n}ell{l}mp{abs(m)}": df_all[f"n{n}ell{l}m0"] + df_all[f"n{n}ell{l}dfreq"]})

        try:
            df_t = df_all[cols].copy()
        except KeyError:
            df_all['run_time'] = 0
            df_t = df_all[cols].copy()

        df_t['param_value'] = df_t.param.apply(lambda x: list(eval(x).values())[param_idx])
        df_t.to_feather(f"../minisauruses/{param}_{rot}.feather")

    cols0 = [f for f in df_t.columns.values if "ell0m" in f]
    cols1 = [f for f in df_t.columns.values if "ell1m" in f]
    cols2 = [f for f in df_t.columns.values if "ell2m" in f]
    cols3 = [f for f in df_t.columns.values if "ell3m" in f]
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

def add_colorbar(fig, ax, params, param_str, ref, palette):
    if isinstance(params[0], float) or isinstance(params[0], int):
        Z = [[0,0],[0,0]]
        levels = np.array(sorted(params)+[params[-1]+np.diff(params)[0]])
        contour = plt.contourf(Z, levels, cmap=mpl.colors.ListedColormap(palette))
        level_step = np.diff(levels)[0]/2
        cb = fig.colorbar(contour, ticks=levels+level_step, ax=ax)
        cb.set_label(rf'{param_str}', fontsize=22, weight='bold')
        cb.set_ticklabels([f"{level:.3f}" for level in levels], fontsize=16)
        if isinstance(ref, int):
            cb.ax.hlines(params[ref]+level_step, 0, 1, color='white', linestyle='--', linewidth=2, alpha=0.8)
        elif isinstance(ref, list):
            for r in ref:
                cb.ax.hlines(params[r]+level_step, 0, 1, color='white', linestyle='--', linewidth=2, alpha=0.8)
        # cb.ax.hlines(params[ref]+level_step, 0, 1, color='white', linestyle='--', linewidth=2, alpha=0.8)
    elif isinstance(params[0], str):
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=1), cmap=mpl.colors.ListedColormap(palette)), 
                            ax=ax, ticks=np.linspace(0, 1, len(params)+1)[:-1]+0.5/(len(params)+1))
        cb.set_label(rf'{param_str}', fontsize=22, weight='bold')
        cb.set_ticklabels(params, fontsize=16)
        if isinstance(ref, int):
            cb.ax.hlines((np.linspace(0, 1, len(params)+1)[:-1]+0.5/(len(params)+1))[ref], 0, 1, color='white', linestyle='--', linewidth=2, alpha=0.8)
        elif isinstance(ref, list):
            for r in ref:
                cb.ax.hlines((np.linspace(0, 1, len(params)+1)[:-1]+0.5/(len(params)+1))[r], 0, 1, color='white', linestyle='--', linewidth=2, alpha=0.8)
        cb.ax.hlines((np.linspace(0, 1, len(params)+1)[:-1]+0.5/(len(params)+1))[ref], 0, 1, color='white', linestyle='--', linewidth=2, alpha=0.8)
    return fig, ax, cb

def plot_fdf(fig, ax, df_master, m, z, v, age, params, param_name='param',
                                param_str='param', ref=0, use_linestyles=False, transparent=False, bgcolor='white'):
    """
    Plots the frequency differences for given parameters.
    """
    patterns = ['ell0m', 'ell1m', 'ell2m', 'ell3m']
    cols = get_filtered_columns(df_master, patterns)
    
    palette, linestyles = configure_plot_style(use_linestyles, params, transparent, bgcolor)
    markers = ['o', '^', 's', 'd']
    
    # Configure transparency and background if needed
    if transparent:
        ax.set_facecolor(bgcolor)
        ax.grid(True, color=bgcolor, linestyle='--', linewidth=1, alpha=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor(bgcolor)

    l_max = 3
    min_val, max_val = 0, 0
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
        ff_cols = [f'n{n}ell{l}m0_ff' for n in range(1, 11) for l in range(l_max+1)]
        min_val = min(min_val, 100*df[ff_cols].min().min())
        max_val = max(max_val, 100*df[ff_cols].max().max())
    # df_this = df.query(f"param_value=={0.014}")
    # this_mean = 100*np.mean(df_this[ff_cols].values)
    # ax.plot(range(4, 116),  np.repeat(this_mean, len(range(4, 116))), lw=2, color='black', zorder=3)
    # ax.text(88, 0.9, r'$\bf{\langle \delta {f/f} \rangle}$'+f'={this_mean:.3f}', size=20, weight='bold', color='black', zorder=3)

    lim = max(abs(min_val), abs(max_val))
    ax.set_ylim(-1.1*lim, 1.1*lim)
    ax.set_xlim(5, 115)


    ax.set_title(f'Age: {age:.2f} Myrs', fontsize=25, weight='bold')
    ax.set_xlabel(r"$\bf{f, \ \ \rm{d}^{-1}}$", size=25, weight='bold')
    ax.set_ylabel(r'$\bf{\delta f/f}$ (%)', size=25, weight='bold')
    
    ax.axhline(0, ls='dashed', color='grey')
    ax.tick_params(axis='both', which='major', labelsize=20)
        

    handles, labels = ax.get_legend_handles_labels()
    unique = [(handle, label) for i, (handle, label) in enumerate(zip(handles, labels)) if label not in labels[:i]]
    fig.legend(*zip(*unique), loc='upper right', bbox_to_anchor=(0.75, 0.88), title=r'$\ell$', prop={'size':15}, title_fontsize=20, ncol=2)

    fig, ax, cb = add_colorbar(fig, ax, params, param_str, ref, palette)
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
        return update_fdf_for_current_age(ax, df_master, age, m, z, v, params, ref, palette, markers, xlim, ylim)
    
    d_age = ages[1] - ages[0] 
    ages = np.arange(ages[0], ages[-1]+d_age, d_age*skipstep)
    ani = mpl.animation.FuncAnimation(fig, update, frames=ages, repeat=True)
    return ani


def update_fdf_for_current_age(ax, df_master, age, m, z, v, params, ref, palette, markers, xlim, ylim):
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


def interactive_fdf(df_master, m, z, v, ages, params, param_name, param_str, ref, use_linestyles=False, transparent=False, bgcolor='white', xlim=(5, 115), ylim=(-5, 5)):
    age_min, age_max = min(ages), max(ages)
    d_age = ages[1] - ages[0]

    palette, linestyles = configure_plot_style(use_linestyles, params, transparent, bgcolor)
    markers = ['o', '^', 's', 'd']

    def plot_with_age(age):
        fig, ax = plt.subplots(figsize=(14, 7))
        update_fdf_for_current_age(ax, df_master, age, m, z, v, params, ref, palette, markers, xlim, ylim)
        fig, ax, cb = add_colorbar(fig, ax, params, param_str, ref, palette)
        plt.show()

    age_slider = FloatSlider(value=age_min, min=age_min, max=age_max, step=d_age, description='Age:')
    
    # Use interactive to create a widget for the age slider
    interactive_plot = interactive(plot_with_age, age=age_slider)
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

    if isinstance(params[0], float) or isinstance(params[0], int):
        Z = [[0,0],[0,0]]
        levels = np.array(sorted(params)+[params[-1]+np.diff(params)[0]])
        contour = plt.contourf(Z, levels, cmap=mpl.colors.ListedColormap(palette))
        level_step = np.diff(levels)[0]/2
        cb = fig.colorbar(contour, ticks=levels+level_step, ax=ax)
        cb.set_label(rf'{param_str}', fontsize=22, weight='bold')
        cb.set_ticklabels([f"{level:.3f}" for level in levels], fontsize=16)
        if isinstance(ref, int):
            cb.ax.hlines(params[ref]+level_step, 0, 1, color='white', linestyle='--', linewidth=2, alpha=0.8)
        elif isinstance(ref, list):
            for r in ref:
                cb.ax.hlines(params[r]+level_step, 0, 1, color='white', linestyle='--', linewidth=2, alpha=0.8)
        # cb.ax.hlines(params[ref]+level_step, 0, 1, color='white', linestyle='--', linewidth=2, alpha=0.8)
    elif isinstance(params[0], str):
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=1), cmap=mpl.colors.ListedColormap(palette)), 
                            ax=ax, ticks=np.linspace(0, 1, len(params)+1)[:-1]+0.5/(len(params)+1))
        cb.set_label(rf'{param_str}', fontsize=22, weight='bold')
        cb.set_ticklabels(params, fontsize=16)
        if isinstance(ref, int):
            cb.ax.hlines((np.linspace(0, 1, len(params)+1)[:-1]+0.5/(len(params)+1))[ref], 0, 1, color='white', linestyle='--', linewidth=2, alpha=0.8)
        elif isinstance(ref, list):
            for r in ref:
                cb.ax.hlines((np.linspace(0, 1, len(params)+1)[:-1]+0.5/(len(params)+1))[r], 0, 1, color='white', linestyle='--', linewidth=2, alpha=0.8)
        # cb.ax.hlines((np.linspace(0, 1, len(params)+1)[:-1]+0.5/(len(params)+1))[ref], 0, 1, color='white', linestyle='--', linewidth=2, alpha=0.8)
    return fig, ax, cb