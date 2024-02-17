from mesaport import ProjectOps
import glob
import tarfile
import os
import argparse
import pandas as pd
import numpy as np
import shutil


def check_if_done(archive_dir, track):
    '''
    Check if GYRE has already been run on the given archive.
    '''
    if os.path.exists(f"{archive_dir}/gyre/freqs_{track}.tar.gz"):
        return True
    else:
        return False

def untar_profiles(profile_tar, jobfs=None):
    """
    Untar all the profiles from a tarball into the jobfs directory.
    """
    if jobfs is None:
        HOME = os.environ["HOME"]
        grid_name = profile_tar.split('/')[-3].split('grid_archive_')[-1]
        try:
            jobfs = os.path.join(os.environ["PBS_JOBFS"], f"gridwork_{grid_name}")
        except KeyError:
            jobfs = os.path.join(os.environ["TMPDIR"], f"gridwork_{grid_name}")
        else:
            jobfs = os.path.abspath(f"./gridwork_{grid_name}")
    if os.path.exists(jobfs):
        jobfs = os.path.abspath(jobfs)
    else:
        os.makedirs(jobfs)
    if not os.path.exists(jobfs):
        raise FileNotFoundError(f"Jobfs directory {jobfs} not found")
    with tarfile.open(profile_tar, 'r:gz') as tar:
        tar.extractall(path=jobfs)
    return os.path.join(jobfs, profile_tar.split('/')[-1].split('.tar.gz')[0])

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
    archive_name = os.path.abspath(archive_name)
    if suffix == None:
        histfile = f"{archive_name}/histories/history.data"
        pindexfile = f"{archive_name}/profile_indexes/profiles.index"
    else:
        histfile = f"{archive_name}/histories/history_{suffix}.data"
        pindexfile = f"{archive_name}/profile_indexes/profiles_{suffix}.index"
    h = pd.read_csv(histfile, delim_whitespace=True, skiprows=5)
    h["Zfrac"] = 1 - h["average_h1"] - h["average_he4"]
    h["Myr"] = h["star_age"]*1.0E-6
    h["density"] = h["star_mass"]/np.power(10,h["log_R"])**3
    p = pd.read_csv(pindexfile, skiprows=1, names=['model_number', 'priority', 'profile_number'], delim_whitespace=True)
    h = pd.merge(h, p, on='model_number', how='inner')
    gyre_start_age = 1e6
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

def save_gyre_outputs(profiles_dir, archive_dir, suffix):
    '''
    Save the GYRE outputs to a tarball in the archive directory.
    '''
    try: 
        shutil.copy(f"{profiles_dir}/gyre.log", f"{archive_dir}/gyre/gyre_{suffix}.log")
        print("Copied GYRE log file")
    except Exception as e:
        print(e)
        print("Failed to copy GYRE log file")
    freq_files = glob.glob(f"{profiles_dir}/*-freqs.dat")
    if len(freq_files) > 0:
        with tarfile.open(f"{archive_dir}/gyre/freqs_{suffix}.tar.gz", "w:gz") as tar:
            for f in freq_files:
                tar.add(f, arcname=f.split('/')[-1])
    else:
        raise RuntimeError("No GYRE frequency files found")
    
def run_gyre(gyre_in, archive_dir, index, cpu_per_process=1, jobfs=None):
    '''
    Run GYRE on a given archive with MESA profiles.
    '''
    archive_dir = os.path.abspath(archive_dir)
    input_file = os.path.join(archive_dir, "track_inputs.csv")
    inputs = pd.read_csv(input_file)
    track = inputs.iloc[index]['track']
    print(f"Running GYRE for track {track}\n")
    if check_if_done(archive_dir, index):
        print("GYRE already run on this archive\n")
    else:
        zinit = float(track.split('_')[1].split('z')[-1])

        profiles, gyre_input_params = get_gyre_params_archived(archive_dir, suffix=track, zinit=zinit, file_format="GSM", run_on_cool=True)
        # print(gyre_input_params)
        profiles_dir = untar_profiles(profile_tar=os.path.join(archive_dir, 'profiles', f'profiles_{track}.tar.gz'), jobfs=jobfs)

        if len(profiles) == 0:
            raise RuntimeError("No profiles to run GYRE on")
        else:
            print(f"{len(profiles)} profiles found to run GYRE on\n")
            
        os.environ['OMP_NUM_THREADS'] = '1'
        # print(profiles)
        # exit()
        proj = ProjectOps()
        res = proj.runGyre(gyre_in=gyre_in, files=profiles,
                        gyre_input_params=gyre_input_params, wdir=profiles_dir,
                        parallel=True, n_cores=cpu_per_process,
                        data_format="GSM", logging=True)
        if not res:
            raise RuntimeError("GYRE run failed")
        try:
            save_gyre_outputs(profiles_dir, archive_dir, track)
        except Exception as e:
            print(e)
            print("Failed to save GYRE outputs")
        else:
            shutil.rmtree(profiles_dir)
            print("GYRE outputs saved\n")


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
        
