import numpy as np
import os, sys
import pandas as pd
import numpy as np
import tarfile
import argparse
import time
from rich import print

def fit_radial(ts, degree=0):
    n_min, n_max = 5, 9
    try:
        vert_freqs = ts.query('n_g == 0').query(f'l=={degree}').query(f'n_pg>={n_min}').query(f'n_pg<={n_max}')[
            ['n_pg', 'freq']].values
    except:
        vert_freqs = ts.query(f'l_obs=={degree}').query(f'n_obs>={n_min}').query(f'n_obs<={n_max}')[
            ['n_obs', 'f_obs']].values
    if len(vert_freqs > 0):
        slope, intercept = np.polyfit(vert_freqs[:, 0], vert_freqs[:, 1], 1)
        r_value, p_value, std_err = 0, 0, 0
    else:
        slope, intercept, r_value, p_value, std_err = np.zeros(5)
    return len(vert_freqs), slope, intercept, r_value, p_value, std_err


def epsilon(ts):
    length_rad, slope, intercept, r_value, p_value, std_err = fit_radial(ts, degree=0)
    eps = intercept / slope
    if length_rad < 3:
        length_dip, slope, intercept, r_value, p_value, std_err = fit_radial(ts, degree=1)
        if length_dip > length_rad:
            eps = intercept / slope - 0.5
    return np.round(eps, 3)


def model_Dnu(ts):
    length_rad, slope, intercept, r_value, p_value, std_err = fit_radial(ts, degree=0)
    if length_rad < 3:
        length_dip, slope, intercept, r_value, p_value, std_err = fit_radial(ts, degree=1)
        if length_rad > length_dip:
            length_rad, slope, intercept, r_value, p_value, std_err = fit_radial(ts, degree=0)
    Dnu = slope
    return np.round(Dnu, 3)


def process_freqs_file(file, h_master, mode_labels, dfreq_labels, dfreq_rot_labels):
    ## Add empty mode labels
    ## All other straightforward methods of doing this gave performance warnings and fragmented dataframe
    ## Concat works fine
    extra_columns = ['omega', 'R_eq', 'R_polar', 'omega_c', 'Dnu', 'eps']
    nan_array = np.full((len(h_master), len(mode_labels+dfreq_labels+dfreq_rot_labels+extra_columns)), np.nan)
    new_cols_df = pd.DataFrame(nan_array, columns=mode_labels+dfreq_labels+dfreq_rot_labels+extra_columns)
    h_master = pd.concat([h_master, new_cols_df], axis=1)

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
                    for i, label in enumerate(mode_labels):
                        n = int(label.split('n')[-1].split('ell')[0])
                        l = int(label.split('ell')[-1].split('m')[0])
                        m = 0
                        freq = ts.query(f'n_pg=={n} and l=={l} and m=={m}')['freq'].values
                        if len(freq) > 0:
                            h[label] = np.round(freq[0], 6)
                            h[f'n{n}ell{l}dfreq_rot'] = np.round(ts[(ts['n_pg'] == n) & (ts['l'] == l) & (ts['m'] == m)]['dfreq_rot'].values[0], 6)
                    h_final_list.append(h)
    h_final = pd.concat(h_final_list)
    return h_final


def get_gyre_freqs(archive_dir, hist, suffix):
    file = os.path.join(archive_dir, 'gyre', f'freqs_{suffix}.tar.gz')
    l_max = 3
    mode_labels = []
    dfreq_labels = []
    dfreq_rot_labels = []
    for l in range(0, l_max+1):
        for n in range(1, 11):
            mode_labels.append(f'n{n}ell{l}m0')
            dfreq_rot_labels.append(f'n{n}ell{l}dfreq_rot')
            # if l > 0:
            #     dfreq_labels.append(f'n{n}ell{l}dfreq')
    hist = process_freqs_file(file, hist, mode_labels, dfreq_labels, dfreq_rot_labels)
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
    h = pd.read_csv(os.path.join(archive_dir, 'histories', f'history_{suffix}.data'), delim_whitespace=True, skiprows=5)
    h['m'] = m
    h['z'] = z
    h['v'] = v
    h['param'] = inputs.iloc[index]['param'] if 'param' in inputs.columns else 0
    h['tr_num'] = param_idx
    h['Myr'] = h['star_age']*1e-6
    h['teff'] = np.round(np.power(10, h['log_Teff']), 2)
    h['density'] = h['star_mass'] / np.power(10, h['log_R']) ** 3
    profile_index = pd.read_csv(os.path.join(archive_dir, 'profile_indexes', f'profiles_{suffix}.index'), 
                                    skiprows=1, names=['model_number', 'priority', 'profile_number'], delim_whitespace=True)
    h = pd.merge(h, profile_index, on='model_number', how='inner').drop_duplicates()
    return h, suffix


def setup_and_run(archive_dir, index):
    sys.stdout.flush()
    print('Start Date: ', time.strftime("%d-%m-%Y", time.localtime()))
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
