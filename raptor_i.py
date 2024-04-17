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
                    ts = pd.read_fwf(tar.extractfile(member), skiprows=5)
                    ts.drop(columns=['M_star', 'R_star', 'Im(freq)', 'E_norm'], inplace=True)
                    ts.rename(columns={'Re(freq)': 'freq'}, inplace=True)
                    h["Dnu"] = model_Dnu(ts)
                    h["eps"] = epsilon(ts)
                    n_g_list = ts.n_g.unique()[1:]
                    n_pg_list = ts.n_pg.unique()
                    l_max = 3
                    for l in range(0, l_max+1):
                        for n in range(1, 11):
                            if n in n_pg_list:
                                n_pg = n
                                freqs = ts.query(f'n_pg=={n_pg} and l=={l} and m==0').freq.values
                                if len(freqs) > 0:
                                    kwargs1 = {f'n{n_pg}ell{l}m0': np.round(freqs[0], 6)}
                                    h = h.assign(**kwargs1)
                                    kwargs2 = {f'n{n_pg}ell{l}dfreq': lambda x: np.round(ts.query(f'n_pg=={n_pg} and l=={l} and m==0').dfreq_rot.values[0], 6)}
                                    h = h.assign(**kwargs2)
                            if n in n_g_list and l>0:
                                n_g = n
                                freqs = ts.query(f'n_g=={n_g} and l=={l} and m==0').freq.values
                                if len(freqs) > 0:
                                    freqs_m1 = ts.query(f'n_g=={n_g} and l=={l} and m==1').freq.values
                                    if len(freqs_m1) > 0:
                                        kwargs2 = {f'ng{n_g}ell{l}dfreq': lambda x: np.round(freqs_m1[0], 6) 
                                                - np.round(freqs[0], 6)}
                                        h = h.assign(**kwargs2)      
                                        kwargs1 = {f'ng{n_g}ell{l}m0': np.round(freqs[0], 6)}
                                        h = h.assign(**kwargs1)
                    h_final_list.append(h)
    h_final = pd.concat(h_final_list)   
    return h_final

def get_gyre_freqs(archive_dir, hist, suffix):
    file = os.path.join(archive_dir, 'gyre', f'freqs_{suffix}.tar.gz')
    hist = process_freqs_file(file, hist)
    return hist


def get_hist(archive_dir, index):
    input_file = os.path.join(archive_dir, 'track_inputs.csv')
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


def setup_and_run(archive_dir, index):
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
    h.to_csv(os.path.join(archive_dir, 'minisauruses', f'minisaurus_{suffix}.csv'), index=False)
    

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
