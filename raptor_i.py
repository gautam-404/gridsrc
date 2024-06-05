import numpy as np
import os, sys
import pandas as pd
import numpy as np
import tarfile
import argparse
import time
from rich import print

def fit_line(x, y):
    if len(x) > 1:
        slope, intercept = np.polyfit(x, y, 1)
        return len(x), slope, intercept
    else:
        return len(x), 0, 0  # Return zero slope and intercept if not enough points

def fit_radial(ts, degree=0):
    n_min, n_max = 5, 9
    try:
        query_string = f'n_g == 0 and l == {degree} and {n_min} <= n_pg <= {n_max}'
        vert_freqs = ts.query(query_string)[['n_pg', 'freq']].values
    except (KeyError, ValueError):
        query_string = f'l_obs == {degree} and {n_min} <= n_obs <= {n_max}'
        vert_freqs = ts.query(query_string)[['n_obs', 'f_obs']].values
    if len(vert_freqs) > 0:
        return fit_line(vert_freqs[:, 0], vert_freqs[:, 1])
    else:
        return 0, 0, 0  # Not enough data, return zero values


def epsilon(ts):
    length_rad, slope, intercept = fit_radial(ts, degree=0)
    if slope != 0:
        eps = intercept / slope
    else:
        eps = 0
    if length_rad < 3:
        length_dip, slope_dip, intercept_dip = fit_radial(ts, degree=1)
        if length_dip > length_rad and slope_dip != 0:
            eps = intercept_dip / slope_dip - 0.5   # adjusting by -0.5 if dipole is more reliable
    return np.round(eps, 3)

def model_Dnu(ts):
    length_rad, slope, intercept = fit_radial(ts, degree=0)
    if length_rad < 3:
        length_dip, slope_dip, intercept_dip = fit_radial(ts, degree=1)
        if length_dip > length_rad:
            slope = slope_dip   # use dipole if it's more reliable
    return np.round(slope, 3)


def process_freqs_file(file, h_master):
    ## Group by profile number
    grouped_h_master = h_master.groupby('profile_number')
    h_final_list = []
    with tarfile.open(file, 'r:gz') as tar:
        ## Read in the frequencies for each profile 
        for member in tar:
            if member.name.endswith('-freqs.dat'):
                profile_number = int(member.name.split('-freqs.dat')[0].split('profile')[-1])
                if profile_number in grouped_h_master.groups:
                    h = grouped_h_master.get_group(profile_number).copy()
                    with tar.extractfile(member) as file:
                        ts = pd.read_fwf(file, skiprows=5)
                        ts.drop(columns=['M_star', 'R_star', 'Im(freq)', 'E_norm'], inplace=True)
                        ts.rename(columns={'Re(freq)': 'freq'}, inplace=True)
                    h["Dnu"] = model_Dnu(ts)
                    h["eps"] = epsilon(ts)
                    n_pg_list = ts.n_pg.unique()
                    n_pg_list = [n_pg for n_pg in n_pg_list if -11 <= n_pg <= 11]
                    l_max = 3
                    for l in range(0, l_max+1):
                        for n_pg in n_pg_list:
                            freqs = ts.query(f'n_pg=={n_pg} and l=={l} and m==0').freq.values
                            if len(freqs) > 0:
                                if n_pg >= 0:
                                    kwargs1 = {f'n{n_pg}ell{l}m0': np.round(freqs[0], 6)}
                                    h = h.assign(**kwargs1)
                                    kwargs2 = {f'n{n_pg}ell{l}dfreq': lambda x: np.round(ts.query(f'n_pg=={n_pg} and l=={l} and m==0').dfreq_rot.values[0], 6)}
                                    h = h.assign(**kwargs2)
                                if n_pg < 0 and l > 0:
                                    freqs_m1 = ts.query(f'n_pg=={n_pg} and l=={l} and m==1').freq.values
                                    if len(freqs_m1) > 0:
                                        kwargs2 = {f'ng{abs(n_pg)}ell{l}dfreq': lambda x: np.round(freqs_m1[0], 6) 
                                                - np.round(freqs[0], 6)}
                                        h = h.assign(**kwargs2)      
                                        kwargs1 = {f'ng{abs(n_pg)}ell{l}m0': np.round(freqs[0], 6)}
                                        h = h.assign(**kwargs1)
                    h_final_list.append(h)
    h_final = pd.concat(h_final_list)   
    return h_final

def get_gyre_freqs(archive_dir, hist, suffix):
    file = os.path.join(archive_dir, 'gyre', f'freqs_{suffix}.tar.gz')
    hist = process_freqs_file(file, hist)
    return hist


def get_hist(archive_dir, index):
    input_file = os.path.join(archive_dir, 'track_inputs_tmp.csv')
    inputs = pd.read_csv(input_file)
    track = inputs.iloc[index]['track']
    m = float(track.split('_')[0][1:])
    z = float(track.split('_')[1][1:])
    v = int(track.split('_')[2][1:])
    param_idx = 0
    if 'param' in inputs.columns:
        param_idx = int(track.split('_param')[-1].split('.')[0])
        suffix = f'm{m}_z{z}_v{v}_param{param_idx}'
    else:
        suffix = f'm{m}_z{z}_v{v}'
    
    logfn = os.path.join(archive_dir, 'runlogs', f'run_{suffix}.log')
    gyrelogfn = os.path.join(archive_dir, 'gyre', f'gyre_{suffix}.log')
    with open(logfn, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "Total time" in line:
                mesa_run_time = float(line.split(" ")[-2])
    with open(gyrelogfn, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "Total time" in line:
                gyre_run_time = float(line.split(" ")[-2])
    
    h = pd.read_csv(os.path.join(archive_dir, 'histories', f'history_{suffix}.data'), sep='\s+', skiprows=5)
    h['mesa_run_time'] = mesa_run_time
    h['gyre_run_time'] = gyre_run_time
    h['m'] = m
    h['z'] = z
    h['v'] = v
    h['param'] = inputs.iloc[index]['param'] if 'param' in inputs.columns else 0
    h['tr_num'] = param_idx
    h['Myr'] = h['star_age']*1e-6
    h['teff'] = np.round(np.power(10, h['log_Teff']), 2)
    h['density'] = h['star_mass'] / np.power(10, h['log_R']) ** 3
    profile_index = pd.read_csv(os.path.join(archive_dir, 'profile_indexes', f'profiles_{suffix}.index'), 
                                    skiprows=1, names=['model_number', 'priority', 'profile_number'], sep='\s+')
    h = pd.merge(h, profile_index, on='model_number', how='inner').drop_duplicates()
    return h, suffix

def find_stabilization_point(x, y, threshold=0.06, window=100):
    dy = np.gradient(y, x)
    d2y = np.gradient(dy, x)
    d2y_diffs = np.diff(d2y)
    for i in range(len(d2y_diffs)):
        tmp = d2y_diffs[i:i+window]
        if np.all(np.abs(tmp) < threshold) and d2y[i] > 0:
            stabilization_point = x[i]
            break
    return stabilization_point

def change_phase(df):
    old_seq = df.phase_of_evolution.values
    new_seq = old_seq.copy()
    Dnu = df.Dnu.values
    age = df.Myr.values
    mid = age.max()/2
    teff = df.teff.values
    dteff = np.gradient(teff, age)
    stabilization_point = find_stabilization_point(age, Dnu)
    stabilization_index = np.where(age == stabilization_point)[0][0]
    for i in range(0, len(old_seq)):
        if new_seq[i] >1:
            new_seq[i] = 2
        if i > stabilization_index:
            if age[i] <= 50:
                new_seq[i] = 3
            else:
                new_seq[i] = 4
                # if new_seq[i+1]==5:
                if age[i] > mid and dteff[i] > 0:
                    break
    new_seq[i:] = 5
    df['new_phase_of_evolution'] = new_seq
    return df

def setup_and_run(archive_dir, index, for_grid=True):
    sys.stdout.flush()
    print('\nStart Date: ', time.strftime("%d-%m-%Y", time.localtime()))
    print('Start time: ', time.strftime("%H:%M:%S", time.localtime()))

    print(f'Producing minisaurus for track index {index} in archive {archive_dir}\n')
    archive_dir = os.path.abspath(archive_dir)
    
    h, suffix = get_hist(archive_dir, index)
    h = get_gyre_freqs(archive_dir, h, suffix)
    h["omega"] = h["surf_avg_v_rot"]*1000*86400 / (np.power(10, h["log_R"])*6.957E08*2*np.pi)
    h["R_eq"] = np.power(10, h["log_R"])
    h["R_polar"] = h["R_eq"] / (1+0.5*np.power(h["omega"], 2))
    h["omega_c"] = h["omega"] / h["surf_avg_omega_div_omega_crit"]
    h.reset_index(drop=True, inplace=True)
    h.drop(columns=["surf_avg_omega_div_omega_crit", "log_R"], inplace=True)
    # if for_grid:
    #     cols_reqd = ['m', 'z', 'v', 'Myr', 'param', 'phase_of_evolution', 'mesa_run_time', 'gyre_run_time', 'tr_num', 'teff', 'log_L', 'density', 'Dnu', 'eps',  'n1ell0m0', 'n2ell0m0',
    #             'n3ell0m0', 'n4ell0m0', 'n5ell0m0', 'n6ell0m0', 'n7ell0m0', 'n8ell0m0',
    #             'n9ell0m0', 'n10ell0m0', 'n11ell0m0', 'n1ell1m0', 'n2ell1m0', 'n3ell1m0',
    #             'n4ell1m0', 'n5ell1m0', 'n6ell1m0', 'n7ell1m0', 'n8ell1m0', 'n9ell1m0',
    #             'n10ell1m0', 'n1ell2m0', 'n2ell2m0', 'n3ell2m0', 'n4ell2m0', 'n5ell2m0',
    #             'n6ell2m0', 'n7ell2m0', 'n8ell2m0', 'n9ell2m0', 'n10ell2m0', 'ng-1ell2m0',
    #             'n0ell2m0', 'n1ell3m0', 'n2ell3m0', 'n3ell3m0', 'n4ell3m0', 'n5ell3m0',
    #             'n6ell3m0', 'n7ell3m0', 'n8ell3m0', 'n9ell3m0', 'n10ell3m0', 'ng-1ell3m0',
    #             'n0ell3m0', 'ng-3ell3m0', 'ng-2ell3m0', 'n11ell1m0', 'n11ell2m0', 'ng-2ell2m0',
    #             'n11ell3m0', 'ng-4ell1m0', 'ng-3ell1m0', 'ng-2ell1m0', 'ng-1ell1m0',
    #             'ng-4ell2m0', 'ng-3ell2m0', 'ng-4ell3m0', 'omega', 'R_eq', 'R_polar', 'omega_c']
    #     h = h[cols_reqd]
    h.sort_values('Myr', inplace=True)
    h = change_phase(h)
    h['phase_of_evolution'] = h['new_phase_of_evolution']
    h.drop('new_phase_of_evolution', axis=1, inplace=True)
    # h.to_csv(os.path.join(archive_dir, 'minisauruses', f'minisaurus_{suffix}.csv'), index=False)
    h.to_feather(os.path.join(archive_dir, 'minisauruses', f'minisaurus_{suffix}.feather'))
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run parameter tests for MESA',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--index', type=int, default=0, 
                        help='Index of the parameter set to run')
    parser.add_argument('-c', '--cores', type=int, default=1, 
                        help='Number of cores to use per process')
    parser.add_argument('-d', '--directory', type=str, default='grid_archive',
                        help='Name of the directory to save outputs')
    args = parser.parse_args()
    config = vars(args)
    index = config['index']
    cpu_per_process = config['cores']
    archive_dir = config['directory']

    if not os.path.exists(os.path.join(archive_dir, 'minisauruses')):
        os.makedirs(os.path.join(archive_dir, 'minisauruses'))
    setup_and_run(archive_dir, index)
