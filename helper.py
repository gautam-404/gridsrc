import os
import sys
import glob
import shutil
import tarfile
import numpy as np
from rich import progress, live, console, panel, prompt, print
from contextlib import contextmanager
import time


Y_sun_phot = 0.2485 # Asplund+2009
Y_sun_bulk = 0.2703 # Asplund+2009
Z_sun_phot = 0.0134 # Asplund+2009
Z_sun_bulk = 0.0142 # Asplund+2009
Y_recommended = 0.28 # typical acceptable value, according to Joel Ong TSC2 talk.
dY_by_dZ = 1.4
h2_to_h1_ratio = 2.0E-05
he3_to_he4_ratio = 1.66E-04

dt_limit_values = ['burn steps', 'Lnuc', 'Lnuc_cat', 'Lnuc_H', 'Lnuc_He', 'lgL_power_phot', 'Lnuc_z', 'bad_X_sum',
                  'dH', 'dH/H', 'dHe', 'dHe/He', 'dHe3', 'dHe3/He3', 'dL/L', 'dX', 'dX/X', 'dX_nuc_drop', 'delta mdot',
                  'delta total J', 'delta_HR', 'delta_mstar', 'diff iters', 'diff steps', 'min_dr_div_cs', 'dt_collapse',
                  'eps_nuc_cntr', 'error rate', 'highT del Ye', 'hold', 'lgL', 'lgP', 'lgP_cntr', 'lgR', 'lgRho', 'lgRho_cntr',
                  'lgT', 'lgT_cntr', 'lgT_max', 'lgT_max_hi_T', 'lgTeff', 'dX_div_X_cntr', 'lg_XC_cntr', 'lg_XH_cntr', 
                  'lg_XHe_cntr', 'lg_XNe_cntr', 'lg_XO_cntr', 'lg_XSi_cntr', 'XC_cntr', 'XH_cntr', 'XHe_cntr', 'XNe_cntr',
                  'XO_cntr', 'XSi_cntr', 'log_eps_nuc', 'max_dt', 'neg_mass_frac', 'adjust_J_q', 'solver iters', 'rel_E_err',
                  'varcontrol', 'max increase', 'max decrease', 'retry', 'b_****']

def initial_abundances(Zinit):
    """
    Input: Zinit
    Output: Yinit, initial_h1, initial_h2, initial_he3, initial_he4
    """
    dZ = np.round(Zinit - Z_sun_bulk,4)
    dY = dY_by_dZ * dZ
    Yinit = np.round(Y_recommended + dY,4)
    Xinit = 1 - Yinit - Zinit

    initial_h2 = h2_to_h1_ratio * Xinit
    initial_he3= he3_to_he4_ratio * Yinit
    initial_h1 = (1 - initial_h2) * Xinit
    initial_he4= (1 - initial_he3) * Yinit

    return Yinit, initial_h1, initial_h2, initial_he3, initial_he4


def phases_params(initial_mass, Zinit):
    '''
    Input: initial_mass, Zinit
    Output: phases_params
    '''
    Yinit, initial_h1, initial_h2, initial_he3, initial_he4 = initial_abundances(Zinit)

    params = {  'Create Pre-MS Model' :
                    {'initial_mass': initial_mass, 'initial_z': Zinit, 'Zbase': Zinit, 'initial_y': Yinit,
                     'mesh_delta_coeff': 1.25,
                    'initial_h1': initial_h1,'initial_h2': initial_h2, 
                    'initial_he3': initial_he3, 'initial_he4': initial_he4,
                    'create_pre_main_sequence_model': True, 'pre_ms_T_c': 9e5,
                    'set_uniform_initial_composition' : True, 'initial_zfracs' : 6,
                    'set_initial_model_number' : True, 'initial_model_number' : 0,
                    # 'change_net' : True, 'new_net_name' : 'pp_and_hot_cno.net',  
                    'change_net' : True, 'new_net_name' : 'pp_and_cno_extras.net',  
                    'change_initial_net' : False, 'adjust_abundances_for_new_isos' : True,
                    # 'set_rates_preference' : True, 'new_rates_preference' : 2,   ## Not available in newer versions
                    'show_net_species_info' : True, 'show_net_reactions_info' : True,
                    'relax_mass' : True, 'lg_max_abs_mdot' : 6, 'new_mass' : initial_mass,
                    'write_header_frequency': 10, 'history_interval': 1, 'terminal_interval': 10, 'profile_interval': 15,
                    'delta_lgTeff_limit' : 0.005, 'delta_lgTeff_hard_limit' : 0.01,
                    'delta_lgL_limit' : 0.02, 'delta_lgL_hard_limit' : 0.05,
                    'okay_to_reduce_gradT_excess' : True, 'Pextra_factor' : 1.0, 'scale_max_correction':0.1},
                    
                'Pre-MS Evolution' :
                    {'initial_mass': initial_mass, 'initial_z': Zinit, 'relax_mass' : True, 'lg_max_abs_mdot' : 6, 'new_mass' : initial_mass,
                    'Zbase': Zinit, 'change_initial_net' : False, 'show_net_species_info' : False, 'show_net_reactions_info' : False,
                    'mesh_delta_coeff': 1.25,
                    'delta_lgTeff_limit' : 0.00015, 'delta_lgTeff_hard_limit' : 0.0015,
                    'set_uniform_initial_composition' : True, 'initial_zfracs' : 6,
                    'change_net' : True, 'new_net_name' : 'pp_and_cno_extras.net',  
                    'change_initial_net' : False, 'adjust_abundances_for_new_isos' : True,
                    'delta_lgL_limit' : 0.0005, 'delta_lgL_hard_limit' : 0.005,
                    'write_header_frequency': 10, 'history_interval': 4, 'terminal_interval': 10, 'profile_interval': 4},


                'Early MS Evolution' :
                    {'initial_mass': initial_mass, 'initial_z': Zinit, 'relax_mass' : True, 'lg_max_abs_mdot' : 6, 'new_mass' : initial_mass,
                    'Zbase': Zinit, 'change_initial_net' : False, 'show_net_species_info' : False, 'show_net_reactions_info' : False,
                    'mesh_delta_coeff': 1.25,
                    'delta_lgTeff_limit' : 0.00015, 'delta_lgTeff_hard_limit' : 0.0015,
                    'set_uniform_initial_composition' : True, 'initial_zfracs' : 6,
                    'change_net' : True, 'new_net_name' : 'pp_and_cno_extras.net',  
                    'change_initial_net' : False, 'adjust_abundances_for_new_isos' : True,
                    'delta_lgL_limit' : 0.0005, 'delta_lgL_hard_limit' : 0.005,
                    'write_header_frequency': 10, 'history_interval': 4, 'terminal_interval': 10, 'profile_interval': 4},

                'Evolution to TAMS' :
                    {'initial_mass': initial_mass, 'initial_z': Zinit, 'relax_mass' : True, 'lg_max_abs_mdot' : 6, 'new_mass' : initial_mass,
                    'Zbase': Zinit, 'change_initial_net' : False, 'show_net_species_info' : False, 'show_net_reactions_info' : False,
                    'mesh_delta_coeff': 1.25,
                    'delta_lgTeff_limit' : 0.0006, 'delta_lgTeff_hard_limit' : 0.006,
                    'set_uniform_initial_composition' : True, 'initial_zfracs' : 6,
                    'change_net' : True, 'new_net_name' : 'pp_and_cno_extras.net',  
                    'change_initial_net' : False, 'adjust_abundances_for_new_isos' : True,
                    'delta_lgL_limit' : 0.002, 'delta_lgL_hard_limit' : 0.02,
                    'write_header_frequency': 10, 'history_interval': 1, 'terminal_interval': 10, 'profile_interval': 1},

                'Evolution post-MS' :
                    {'initial_mass': initial_mass, 'initial_z': Zinit, 'relax_mass' : True, 'lg_max_abs_mdot' : 6, 'new_mass' : initial_mass,
                    'Zbase': Zinit, 'change_initial_net' : False, 'show_net_species_info' : False, 'show_net_reactions_info' : False,
                    'mesh_delta_coeff': 1.25,
                    'delta_lgTeff_limit' : 0.0006, 'delta_lgTeff_hard_limit' : 0.006,
                    'set_uniform_initial_composition' : True, 'initial_zfracs' : 6,
                    'change_net' : True, 'new_net_name' : 'pp_and_cno_extras.net',  
                    'change_initial_net' : False, 'adjust_abundances_for_new_isos' : True,
                    'delta_lgL_limit' : 0.002, 'delta_lgL_hard_limit' : 0.02,
                    'write_header_frequency': 10, 'history_interval': 1, 'terminal_interval': 10, 'profile_interval': 1},
    }

    return params

def mute():
    sys.stdout = open(os.devnull, 'w') 

def unmute():
    sys.stdout = sys.__stdout__

def read_error(name):
    retry_type = ""
    terminate_type = ""
    with open(f"{name}/run.log", "r") as f:
        for outline in f:
            splitline = outline.split(" ")
            if "retry:" in splitline:
                retry_type = outline.split(" ")
            if "terminated" in splitline and "evolution:" in splitline:
                terminate_type = outline.split(" ")
            if "ERROR" in splitline:
                error_type = outline.split(" ")
                terminate_type = error_type
            if "specified photo does not exist" in outline:
                terminate_type = "photo does not exist"
    print(retry_type, terminate_type)
    return retry_type, terminate_type

@contextmanager
def cwd(path):
    '''
    Change directory to path, then return to original directory.
    '''
    oldpwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)

def archive_LOGS(name, track_index, save_track, gyre, archive_path, tracks_dir="/g/data/qq01/tracks"):
    # path = os.path.abspath(os.getcwd().split("MESA-grid")[0], "MESA-grid"))
    # os.chdir(path)
    shutil.copy(f"{name}/LOGS/history.data", archive_path+f"/histories/history_{track_index}.data")
    shutil.copy(f"{name}/LOGS/profiles.index", archive_path+f"/profile_indexes/profiles_{track_index}.index")
    profiles_dir = os.path.abspath(archive_path+f"/profiles/profiles_{track_index}")
    
    if gyre:
        gyre_archive = os.path.abspath(archive_path+f"/gyre/freqs_{track_index}")
        if not os.path.exists(gyre_archive):
            os.mkdir(gyre_archive)
        for file in glob.glob(os.path.join(name, "LOGS/*-freqs.dat")):
            shutil.copy(file, gyre_archive)
        with tarfile.open(f"{gyre_archive}.tar.gz", "w:gz") as tarhandle:
            tarhandle.add(gyre_archive, arcname=os.path.basename(gyre_archive))
        shutil.rmtree(gyre_archive)
    if save_track:
        if not os.path.exists(profiles_dir):
            os.mkdir(profiles_dir)

        mesa_profiles = glob.glob(os.path.join(name, "LOGS/profile*.data"))
        try:
            gyre_profiles = glob.glob(os.path.join(name, "LOGS/profile*.data.GSM"))
        except:
            gyre_profiles = glob.glob(os.path.join(name, "LOGS/profile*.data.GYRE"))

        for mesa_file, gyre_file in zip(mesa_profiles, gyre_profiles):
            shutil.copy(mesa_file, profiles_dir)
            shutil.copy(gyre_file, profiles_dir)
        with tarfile.open(f"{profiles_dir}.tar.gz", "w:gz") as tarhandle:
            tarhandle.add(profiles_dir, arcname=os.path.basename(profiles_dir))
        shutil.rmtree(profiles_dir)
        # if not os.path.exists(tracks_dir):
        #     os.mkdir(tracks_dir)
        # compressed_file = f"{tracks_dir}/track_{track_index}.tar.gz"
        # with tarfile.open(compressed_file, "w:gz") as tarhandle:
        #     tarhandle.add(name, arcname=os.path.basename(name))
    shutil.rmtree(name)


def create_grid_dirs(overwrite=False, archive_path="grid_archive"):
    '''
    Create grid directories. 
    Args:   overwrite (bool): overwrite existing grid directories.
                            If overwrite is None, prompt user to overwrite existing grid directories.
    '''
    ## Create archive directories
    if overwrite:
        if os.path.exists(archive_path):
            try:
                shutil.rmtree(archive_path)
            except:
                try:
                    os.system(f"rm -rf {archive_path}")
                except:
                    pass
        os.mkdir(archive_path)
        os.mkdir(os.path.join(archive_path, "tracks"))
        os.mkdir(os.path.join(archive_path, "histories"))
        os.mkdir(os.path.join(archive_path, "profile_indexes"))
        os.mkdir(os.path.join(archive_path, "profiles"))
        os.mkdir(os.path.join(archive_path, "gyre"))
        os.mkdir(os.path.join(archive_path, "failed"))
        os.mkdir(os.path.join(archive_path, "runlogs"))
        os.mkdir(os.path.join(archive_path, "inlists"))
    else:
        if not os.path.exists(archive_path):
            os.mkdir(archive_path)
            os.mkdir(os.path.join(archive_path, "tracks"))
            os.mkdir(os.path.join(archive_path, "histories"))
            os.mkdir(os.path.join(archive_path, "profile_indexes"))
            os.mkdir(os.path.join(archive_path, "profiles"))
            os.mkdir(os.path.join(archive_path, "gyre"))
            os.mkdir(os.path.join(archive_path, "failed"))
            os.mkdir(os.path.join(archive_path, "runlogs"))
            os.mkdir(os.path.join(archive_path, "inlists"))

def progress_columns():
    '''Define progress bar columns'''
    progress_columns = (progress.SpinnerColumn(),
                progress.TextColumn("[progress.description]{task.description}"),
                progress.BarColumn(bar_width=60),
                progress.MofNCompleteColumn(),
                progress.TaskProgressColumn(),
                progress.TimeElapsedColumn())
    return progress_columns