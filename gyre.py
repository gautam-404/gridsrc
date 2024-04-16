import os
import shutil
import pandas as pd
import numpy as np
import glob
import tarfile
from itertools import repeat
import time
import traceback
import logging

from mesaport import ProjectOps, MesaAccess
from rich import print

from .pool import ray_pool, mp_pool
from . import helper

def get_gyre_params_archived(track_index, grid_archive, profiles_archive, file_format='GSM'):
    '''
    Compute the GYRE input parameters for a given track index.

    Parameters
    ----------
    track_index : int
        Index of the track
    grid_archive : str
        Path to the grid archive.
    profiles_archive : str
        Path to the profiles archive.

    Returns
    -------
    profiles : list
        List of MESA profile files to be run with GYRE.
    gyre_input_params : list
    '''
    histfile = os.path.join(grid_archive, f"histories/history_{track_index}.data")
    pindexfile = os.path.join(grid_archive, f"profile_indexes/profiles_{track_index}.index")
    h = pd.read_csv(histfile, sep='\s+', skiprows=5)
    p = pd.read_csv(pindexfile, skiprows=1, names=['model_number', 'priority', 'profile_number'], sep='\s+')
    h = pd.merge(h, p, on='model_number', how='right')
    h["Zfrac"] = 1 - h["average_h1"] - h["average_he4"]
    h["Myr"] = h["star_age"]*1.0E-6
    h["density"] = h["star_mass"]/np.power(10,h["log_R"])**3
    gyre_start_age = 2.0E6
    gyre_intake = h.query(f"Myr > {gyre_start_age/1.0E6}")
    profiles = []
    gyre_input_params = []
    zinit = None
    run_on_cool = False
    for i,row in gyre_intake.iterrows():
        p = int(row["profile_number"])
        # profile_file = f"{name}/LOGS/profile{p}.data.{file_format}"
        profile_file = os.path.join(profiles_archive, f"profile{p}.data.{file_format}")
        if zinit == None:
            zinit = pd.read_csv(profile_file, header=1, nrows=1, sep='\s+')['initial_z'].values[0]
            
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
            freq_min = int(1.5 * dnu)
            freq_max = int(12 * dnu)
        except:
            dnu = None
            freq_min = 15
            if zinit < 0.003:
                freq_max = 150
            else:
                freq_max = 95
        profiles.append(profile_file)
        gyre_input_params.append({"freq_min": freq_min, "freq_max": freq_max, "diff_scheme": diff_scheme})
    return profiles, gyre_input_params


def get_gyre_params_archived(archive_name, suffix=None, zinit=None, run_on_cool=False, file_format="GYRE"):
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
    archive_name= os.path.abspath(archive_name)
    if suffix == None:
        histfile = f"{archive_name}/histories/history.data"
        pindexfile = f"{archive_name}/profile_indexes/profiles.index"
    else:
        histfile = f"{archive_name}/histories/history_{suffix}.data"
        pindexfile = f"{archive_name}/profile_indexes/profiles_{suffix}.index"
    h = pd.read_csv(histfile, sep='\s+', skiprows=5)
    p = pd.read_csv(pindexfile, skiprows=1, names=['model_number', 'priority', 'profile_number'], sep='\s+')
    h = pd.merge(h, p, on='model_number', how='right')
    h["Zfrac"] = 1 - h["average_h1"] - h["average_he4"]
    h["Myr"] = h["star_age"]*1.0E-6
    h["density"] = h["star_mass"]/np.power(10,h["log_R"])**3
    gyre_start_age = 2.0E6
    gyre_intake = h.query(f"Myr > {gyre_start_age/1.0E6}")
    profiles = []
    gyre_input_params = []
    for i,row in gyre_intake.iterrows():
        p = int(row["profile_number"])
        if suffix == None:
            mesa_profile = f"{archive_name}/LOGS/profile{p}.data"
            gyre_profile = f"{archive_name}/LOGS/profile{p}.data.{file_format}"
        else:
            mesa_profile = f"{archive_name}/profiles/profiles_{suffix}/profile{p}.data"
            gyre_profile = f"{archive_name}/profiles/profiles_{suffix}/profile{p}.data.{file_format}"
        if zinit == None:
            zinit = pd.read_csv(mesa_profile, header=1, nrows=1, sep='\s+')['initial_z'].values[0]
            
        ###Checks###
        # if not os.path.exists(profile_file):
        #     raise FileNotFoundError("Profile not found. Possible file format mismatch")
        # if row["log_Teff"] < 3.778 and not run_on_cool:
        #     continue
        ############

        if row["Myr"] > 40:
            diff_scheme = "COLLOC_GL2"
        else:
            diff_scheme = "MAGNUS_GL2"
        try:
            muhz_to_cd = 86400/1.0E6
            mesa_dnu = row["delta_nu"]
            dnu = mesa_dnu * muhz_to_cd
            freq_min = int(1.5 * dnu)
            freq_max = int(12 * dnu)
        except:
            dnu = None
            freq_min = 15
            if zinit < 0.003:
                freq_max = 150
            else:
                freq_max = 95
        profiles.append(gyre_profile)
        gyre_input_params.append({"freq_min": freq_min, "freq_max": freq_max, "diff_scheme": diff_scheme})
    return profiles, gyre_input_params


def get_gyre_params(name, zinit=None, run_on_cool=False, file_format="GYRE"):
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
    histfile = f"{name}/LOGS/history.data"
    pindexfile = f"{name}/LOGS/profiles.index"
    h = pd.read_csv(histfile, sep='\s+', skiprows=5)
    p = pd.read_csv(pindexfile, skiprows=1, names=['model_number', 'priority', 'profile_number'], sep='\s+')
    h = pd.merge(h, p, on='model_number', how='right')
    h["Zfrac"] = 1 - h["average_h1"] - h["average_he4"]
    h["Myr"] = h["star_age"]*1.0E-6
    h["density"] = h["star_mass"]/np.power(10,h["log_R"])**3
    gyre_start_age = 2.0E6
    gyre_intake = h.query(f"Myr > {gyre_start_age/1.0E6}")
    profiles = []
    gyre_input_params = []
    for i,row in gyre_intake.iterrows():
        p = int(row["profile_number"])
        profile_file = f"{name}/LOGS/profile{p}.data.{file_format}"
        if zinit == None:
            zinit = pd.read_csv(profile_file, header=1, nrows=1, sep='\s+')['initial_z'].values[0]
            
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
            freq_min = int(1.5 * dnu)
            freq_max = int(12 * dnu)
        except:
            dnu = None
            freq_min = 15
            if zinit < 0.003:
                freq_max = 150
            else:
                freq_max = 95
        profiles.append(profile_file)
        gyre_input_params.append({"freq_min": freq_min, "freq_max": freq_max, "diff_scheme": diff_scheme})
    return profiles, gyre_input_params


def gyre_parallel(args):
    '''
    Run GYRE on all .tar.gz tracks within a directory or from a list of .tar.gz track paths.

    Parameters
    ----------
    track : str or list
        Path to .tar.gz track.
    target_dir : str
        Path to directory where GYRE output will be saved.
    gyre_in : dict
        Dictionary of GYRE input parameters.
    jobfs : str
        Path to jobfs directory.
    n_cores : int
        Number of cores to use. The default is 1 thread for each call of this function.
    '''
    track, target_dir, gyre_in, jobfs, n_cores = args
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    name = track.split('/')[-1].split('.')[0]
    index = int(name.split('_')[-1])
    gyre_save_path = os.path.abspath(os.path.join(target_dir, f"freqs_{index}"))
    print("Saving GYRE output to", gyre_save_path)
    ## save freqs data
    if not os.path.exists(gyre_save_path):
        os.mkdir(gyre_save_path)
    with open(f"{gyre_save_path}/gyre.log", "w+") as f:
        f.write(f"Running GYRE on track {index}\n")
    gyre_flag = False
    try:
        with tarfile.open(track, "r:gz") as tar:
            tar.extractall(path=jobfs)
        with helper.cwd(jobfs):
            if not os.path.exists(name):
                name = name.replace("track_", "work_")
                if not os.path.exists(name):
                    raise FileNotFoundError(f"Could not find {name} in {jobfs}")
                else:
                    print(f"[b][blue]Running GYRE on[/blue] {name}")
                    ## Get GYRE params
                    data_format = "GSM"
                    profiles, gyre_input_params = get_gyre_params(name, file_format=data_format)
                    if len(profiles) > 0:
                        if not os.path.exists(profiles[0]):
                            data_format = "GYRE"
                            profiles = [profiles.split("/")[-1].replace("GSM", "GYRE") for profiles in profiles]
                        else:
                            profiles = [profiles.split("/")[-1] for profiles in profiles]
                        # Run GYRE
                        proj = ProjectOps(name)
                        os.environ['OMP_NUM_THREADS'] = '1'
                        ## Run GYRE on multiple profile files parallely
                        proj.runGyre(gyre_in=gyre_in, files=profiles, gyre_input_params=gyre_input_params, 
                                    data_format=data_format, logging=False, parallel=True, n_cores=n_cores, logdir=f"{gyre_save_path}/gyre.log")
                        gyre_flag = True
                    else:
                        with open(f"{name}/run.log", "a+") as f:
                            f.write(f"GYRE skipped: no profiles found, possibly because models where T_eff < 6000 K\n")
    except Exception as e:
        gyre_flag = False
        print(f"[b][red]Error running GYRE on[/red] {name}")
        logging.error(traceback.format_exc())
        with open(f"{gyre_save_path}/gyre.log", "a+") as f:
            f.write(f"Error running GYRE on {name}\n")
            f.write(traceback.format_exc())
        # raise e
        print(e)
    finally:
        if gyre_flag:
            for file in glob.glob(f"{name}/LOGS/*-freqs.dat"):
                shutil.move(file, gyre_save_path)
            ## compress GYRE output
            compressed_file = f"{gyre_save_path}.tar.gz"
            with tarfile.open(compressed_file, "w:gz") as tar:
                tar.add(gyre_save_path, arcname=os.path.basename(gyre_save_path))

        ## Remove GYRE output
        shutil.rmtree(gyre_save_path)
        ## Remove work directory
        i = 1
        while os.path.exists(os.path.join(jobfs, name)): ## Try a few times, then give up. NFS is weird. Gotta wait and retry.
            os.system(f"rm -rf {os.path.join(jobfs, name)} > /dev/null 2>&1")
            time.sleep(0.5)                       ## Wait for the process, that has the nfs files open, to die/diconnect
            if i>5:
                break
            


    
def run_GYRE(tracks_path, target_dir, gyre_in, **kwargs):
    '''
    run GYRE on all .tar.gz tracks within a directory or from a list of .tar.gz track paths.

    Parameters
    ----------
    tracks_path : str or list
        Path to directory containing .tar.gz tracks.
        Or list of paths to .tar.gz tracks.
    target_dir : str
        Path to directory where GYRE output will be saved.
    gyre_in : dict
        Dictionary of GYRE input parameters.
    cpu_per_process : int, optional
        CPU cores per process. The default is 16.
    use_ray : bool, optional
        If True, use ray for multiprocessing. The default is False.
    skip_done : bool, optional
        Default is True. Skip tracks that have already been processed
    slice_start : int, optional
        Default is 0. Required when you want to divide the tracks into segments to be run on different machines/nodes. slice start = first track_index
    slice_end : int, optional
        Default is a random large number, 100000000. Required when you want to divide the tracks into segments to be run on different machines/nodes.
    '''
    cpu_per_process, use_ray, skip_done = kwargs.get("cpu_per_process", 2), kwargs.get("use_ray", False), kwargs.get("skip_done", True)
    slice_start, slice_end = kwargs.get("slice_start", 1), kwargs.get("slice_end", 100000000)

    if isinstance(tracks_path, list):
        tracks = tracks_path
    elif os.path.isdir(tracks_path):
        tracks = glob.glob(os.path.join(tracks_path, "track_*.tar.gz"))
    
    # Run GYRE for only the selected slice of tracks
    tracks = [track for track in tracks if slice_start <= int(track.split('track_')[-1].split('.')[0]) <= slice_end]

    target_dir = os.path.abspath(target_dir)

    ## Skip tracks that ave already been processed
    if skip_done:
        done = glob.glob(f"{target_dir}/freqs_*.tar.gz")
        done = [int(file.split('freqs_')[-1].split('.')[0]) for file in done]
        tracks = [track for track in tracks if int(track.split('track_')[-1].split('.')[0]) not in done]

    gyre_in = os.path.abspath(gyre_in)
    try:
        jobfs = os.environ["PBS_JOBFS"]
    except KeyError:
        jobfs = "./gridwork_gyre"
    
    args = zip(tracks, repeat(target_dir), repeat(gyre_in), repeat(jobfs), repeat(cpu_per_process))
    if use_ray:
        ray_pool(gyre_parallel, args, len(tracks), cpu_per_process=cpu_per_process)
    else:
        mp_pool(gyre_parallel, args, len(tracks), cpu_per_process=cpu_per_process)


    

