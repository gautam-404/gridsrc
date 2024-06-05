from mesaport import ProjectOps
import glob
import tarfile
import os, sys
import argparse
import pandas as pd
import numpy as np
import shutil
import time
import platform
from rich import print
from pathlib import Path

from . import helper


def check_if_done(archive_dir, track, profiles):
    '''
    Check if GYRE has already been run on the given archive.
    '''
    if os.path.exists(os.path.join(archive_dir, "gyre", f"freqs_{track}.tar.gz")):
        try:
            with tarfile.open(os.path.join(archive_dir, "gyre", f"freqs_{track}.tar.gz"), "r:gz") as tar:
                if len(tar.getnames()) > 0 and all([p in tar.getnames() for p in profiles]):
                    return True
                else:
                    return False
        except:
            print("Error reading previously saved freqs tar")
            return False
    else:
        return False

def get_done_profile_idxs(archive_dir, track):
    '''
    Get the profile indexes for which GYRE has already been run.
    '''
    if os.path.exists(os.path.join(archive_dir, "gyre", f"freqs_{track}.tar.gz")):
        try:
            with tarfile.open(os.path.join(archive_dir, "gyre", f"freqs_{track}.tar.gz"), "r:gz") as tar:
                return sorted([int(p.split('-')[0].split('profile')[-1]) for p in tar.getnames()])
        except:
            print("Error reading previously saved freqs tar")
            return []
    else:
        return []

def untar_profiles(profiles_tar, track, jobfs=None):
    """
    Untar all the profiles from a tarball into the jobfs directory.
    """
    print("Untarring profile files")
    if jobfs is None:
        HOME = os.environ["HOME"]
        # grid_name = profile_tar.split('/')[-3].split('grid_archive_')[-1]
        grid_name = profiles_tar.split('/')[-3].split('grid_')[-1]
        if "macOS" in platform.platform():
            jobfs = os.path.abspath(f"./gridwork_{grid_name}")
        else:
            try:
                jobfs = os.path.join(os.environ["PBS_JOBFS"], f"gridwork")
            except KeyError:
                jobfs = os.path.join(os.environ["TMPDIR"], f"gridwork")
            except Exception as e:
                print(e)
                jobfs = os.path.abspath(f"./gridwork_{grid_name}")
    if os.path.exists(jobfs):
        jobfs = os.path.abspath(jobfs)
    else:
        os.makedirs(jobfs)
    if not os.path.exists(jobfs):
        raise FileNotFoundError(f"Jobfs directory {jobfs} not found")
    
    profiles_dir = os.path.join(jobfs, f'profiles_{track}')
    with tarfile.open(profiles_tar, 'r:gz') as tar:
        members = [m for m in tar.getmembers() if '.GSM' in m.name or '.GYRE' in m.name]
        if sys.version_info[1] >= 12: ## python 3.12
            tar.extractall(path=jobfs, members=members, filter='data')
        else:
            tar.extractall(path=jobfs, members=members)
    return profiles_dir

def get_gyre_params(archive_name, suffix=None, zinit=None, run_on_cool=False, file_format="GYRE"):
    '''
    Compute the GYRE input parameters for a given MESA model.

    Parameters
    ----------
    name : str
        Name of the MESA model.
    zinit : float, optional
        Initial metallicity of the MESA model. The default is None. If None, the metallicity is read from the MESA model.
    run_on_cool : bool, optional
        If True, run GYRE on all models, regardless of temperature. The default is False.
    file_format : str, optional
        File format of the MESA model. The default is "GYRE".
    
    Returns
    -------
    profiles : list
        List of MESA profile files to be run with GYRE.
    gyre_input_params : list
    '''
    archive_name = os.path.abspath(archive_name)
    if suffix == None:
        histfile = os.path.join(archive_name, "histories", "history.data")
        pindexfile = os.path.join(archive_name, "profile_indexes", "profiles.index")
    else:
        histfile = os.path.join(archive_name, "histories", f"history_{suffix}.data")
        pindexfile = os.path.join(archive_name, "profile_indexes", f"profiles_{suffix}.index")
    h = pd.read_csv(histfile, sep='\s+', skiprows=5)
    h["Zfrac"] = 1 - h["average_h1"] - h["average_he4"]
    h["Myr"] = h["star_age"]*1.0E-6
    h["density"] = h["star_mass"]/np.power(10,h["log_R"])**3
    p = pd.read_csv(pindexfile, skiprows=1, names=['model_number', 'priority', 'profile_number'], sep='\s+')
    h = pd.merge(h, p, on='model_number', how='inner')
    gyre_start_age = 0
    gyre_intake = h.query(f"Myr > {gyre_start_age/1e6}")
    profiles = []
    gyre_input_params = []
    for i,row in gyre_intake.iterrows():
        p = int(row["profile_number"])
        gyre_profile = f"profile{p}.data.{file_format}"
            
        ###Checks###
        # if not os.path.exists(profile_file):
        #     raise FileNotFoundError("Profile not found. Possible file format mismatch")
        if row["log_Teff"] < 3.778 and not run_on_cool:
            continue
        ############

        if row["Myr"] > 40:
            diff_scheme = "COLLOC_GL2"
        else:
            diff_scheme = "MAGNUS_GL2"
        try:
            muhz_to_cd = 86400/1.0E6
            mesa_dnu = row["delta_nu"]
            dnu = mesa_dnu * muhz_to_cd
            # freq_min = int(1.5 * dnu)
            freq_min = 3
            freq_max = int(12 * dnu)
        except:
            dnu = None
            freq_min = 3
            if zinit < 0.003:
                freq_max = 150
            else:
                freq_max = 95
        profiles.append(gyre_profile)
        gyre_input_params.append({"freq_min": freq_min, "freq_max": freq_max, "diff_scheme": diff_scheme})
    return profiles, gyre_input_params

def save_gyre_outputs(profiles_dir, archive_dir, suffix):
    '''
    Save the GYRE outputs to a tarball in the archive directory.
    '''
    if os.path.exists(os.path.join(archive_dir, "gyre", f"freqs_{suffix}.tar.gz")):
        with tarfile.open(os.path.join(archive_dir, "gyre", f"freqs_{suffix}.tar.gz"), "r:gz") as tar:
            tar.extractall(path=profiles_dir)
    freq_files = [str(p) for p in Path(profiles_dir).rglob("*-freqs.dat") if p.is_file()]
    if len(freq_files) > 0:
        with tarfile.open(os.path.join(archive_dir, "gyre", f"freqs_{suffix}.tar.gz"), "w:gz") as tar:
            for f in freq_files:
                tar.add(f, arcname=f.split('/')[-1])
    else:
        raise RuntimeError("No GYRE frequency files found")
    
def run_gyre(gyre_in, archive_dir, index, cpu_per_process=1, jobfs=None, file_format="GSM"):
    '''
    Run GYRE on a given archive with MESA profiles.
    '''
    sys.stdout.flush()
    print('\nStart Date: ', time.strftime("%d-%m-%Y", time.localtime()))
    print('Start time: ', time.strftime("%H:%M:%S", time.localtime()))

    archive_dir = os.path.abspath(archive_dir)
    input_file = os.path.join(archive_dir, "track_inputs.csv")
    inputs = pd.read_csv(input_file)
    track = inputs.iloc[index]['track']
    print(f"Running GYRE for track {track}\n")

    zinit = float(track.split('_')[1].split('z')[-1])
    profiles, gyre_input_params = get_gyre_params(archive_dir, suffix=track, zinit=zinit, file_format=file_format, run_on_cool=True)
    if len(profiles) == 0:
        raise RuntimeError("No profiles to run GYRE on")
    else:
        print(f"{len(profiles)} profiles found\n")

    if check_if_done(archive_dir, track, profiles):
        print("GYRE already run on this archive\n")
    else:
        profiles_dir = untar_profiles(profiles_tar=os.path.join(archive_dir, 'profiles', f'profiles_{track}.tar.gz'), track=track, jobfs=jobfs)
        done_profiles = get_done_profile_idxs(archive_dir, track)
        print(f"{len(done_profiles)} profiles already done\n")
        profiles = [p for i,p in enumerate(profiles) if i+1 not in done_profiles]
        gyre_input_params = [p for i,p in enumerate(gyre_input_params) if i+1 not in done_profiles]
        profiles_idx = sorted([int(p.split('-')[0].split('profile')[-1].split('.')[0]) for p in profiles])

        chunks = 50
        start_time = time.time()
        profile_chunks = [profiles[i:i+50] for i in range(0, len(profiles_idx), chunks)]
        gyre_input_params_chunks = [gyre_input_params[i:i+50] for i in range(0, len(profiles_idx), chunks)]
        for i, p_chunk in enumerate(profile_chunks):
            print(f'Running GYRE on profiles {p_chunk[0]} to {p_chunk[-1]}')
            os.environ['OMP_NUM_THREADS'] = '1'
            with helper.cwd('.'):
                proj = ProjectOps()
                res = proj.runGyre(gyre_in=gyre_in, files=p_chunk,
                                gyre_input_params=gyre_input_params_chunks[i], wdir=profiles_dir,
                                parallel=True, n_cores=cpu_per_process,
                                data_format=file_format, logging=True)
            if not res:
                raise RuntimeError("GYRE run failed")
            try:
                save_gyre_outputs(profiles_dir, archive_dir, track)
                print(f"GYRE outputs saved\n")
            except Exception as e:
                print(e)
                raise RuntimeError("Failed to save GYRE outputs")
        end_time = time.time()

        with open(f"{profiles_dir}/gyre.log", "a+") as f:
            f.write(f"Total time: {end_time-start_time} s\n\n")
        try: 
            shutil.copy(os.path.join(profiles_dir, "gyre.log"), os.path.join(archive_dir, "gyre", f"gyre_{track}.log"))
            print("Copied GYRE log file")
        except Exception as e:
            print(e)
            print("Failed to copy GYRE log file")
        shutil.rmtree(profiles_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run parameter tests for MESA",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--index", type=int, default=0, 
                        help="Index of the parameter set to run")
    parser.add_argument("-c", "--cores", type=int, default=1, 
                        help="Number of cores to use per process")
    parser.add_argument("-d", "--directory", type=str, default="grid_archive",
                        help="Name of the directory to save outputs")
    args = parser.parse_args()
    config = vars(args)
    index = config["index"]
    cpu_per_process = config["cores"]
    archive_dir = config["directory"]

    run_gyre(gyre_in='src/templates/gyre_rot_template_ell3.in', archive_dir=archive_dir, 
             index=index, cpu_per_process=cpu_per_process)
        
