import numpy as np
import pandas as pd
import glob 
import tarfile
import os

def get_star_from_grid(m, z, v, grid='grid_archive'):
    histfile = f'{grid}/histories/history_m{m}_z{z}_v{v}.data'
    pindexfile = f'{grid}/profile_indexes/profiles_m{m}_z{z}_v{v}.index'
    history_df = pd.read_csv(histfile, skiprows=5, delim_whitespace=True)
    profile_index_df = pd.read_csv(pindexfile, skiprows=1, names=['model_number', 'priority', 'profile_number'], delim_whitespace=True)


    unified_df = pd.DataFrame()
    with tarfile.open(f'{grid}/profiles/profiles_m{m}_z{z}_v{v}.tar.gz') as tar:
        profile_files = tar.getnames()
        for _, history_row in history_df.iterrows():
            model_number = history_row['model_number']
            if model_number in profile_index_df['model_number'].values:
                profile_number = profile_index_df[profile_index_df['model_number'] == model_number]['profile_number'].iloc[0]
                pfile_substr = f'profile{profile_number}'
                pfile = [f for f in profile_files if pfile_substr in f][0]
                profile_df = pd.read_csv(tar.extractfile(pfile), skiprows=5, delim_whitespace=True)
                repeated_history_df = pd.DataFrame([history_row] * len(profile_df))

                combined_df = pd.concat([repeated_history_df.reset_index(drop=True), profile_df.reset_index(drop=True)], axis=1)
                unified_df = pd.concat([unified_df, combined_df], axis=0, ignore_index=True)
    return unified_df

def get_star(workdir='work'):
    workdir = os.path.join(os.path.abspath(workdir), 'LOGS')
    histfile = os.path.join(workdir, 'history.data')
    pindex = os.path.join(workdir, 'profiles.index')

    history_df = pd.read_csv(histfile, skiprows=5, delim_whitespace=True)
    profile_index_df = pd.read_csv(pindex, skiprows=1, names=['model_number', 'priority', 'profile_number'], delim_whitespace=True)

    unified_df = pd.DataFrame()
    for _, history_row in history_df.iterrows():
        model_number = history_row['model_number']
        if model_number in profile_index_df['model_number'].values:
            profile_number = profile_index_df[profile_index_df['model_number'] == model_number]['profile_number'].iloc[0]

            profile_file = os.path.join(workdir, f'profile{profile_number}.data')
            if os.path.exists(profile_file):
                profile_df = pd.read_csv(profile_file, skiprows=5, delim_whitespace=True)
                repeated_history_df = pd.DataFrame([history_row] * len(profile_df))

                combined_df = pd.concat([repeated_history_df.reset_index(drop=True), profile_df.reset_index(drop=True)], axis=1)
                unified_df = pd.concat([unified_df, combined_df], axis=0, ignore_index=True)
    return unified_df

