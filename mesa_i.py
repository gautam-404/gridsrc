import os, subprocess
from mesaport import MesaAccess, ProjectOps
import argparse
from itertools import product
import time
import pandas as pd
import shutil

from src import helper, gyre
from rich import print

def teff_helper(star, retry):
    delta_lgTeff_limit = star.get("delta_lgTeff_limit")
    delta_lgTeff_hard_limit = star.get("delta_lgTeff_hard_limit")
    # delta_lgTeff_limit += delta_lgTeff_limit/10
    delta_lgTeff_hard_limit += retry*delta_lgTeff_hard_limit
    star.set({"delta_lgTeff_limit": delta_lgTeff_limit, "delta_lgTeff_hard_limit": delta_lgTeff_hard_limit}, force=True)

def check_if_done(name_og, archive_path):
    archive_path = os.path.abspath(archive_path)
    runlog_present = os.path.exists(archive_path+f"/runlogs/run_{name_og}.log")
    profiles_archived = os.path.exists(archive_path+f"/profiles/profiles_{name_og}.tar.gz") and os.path.exists(archive_path+f"/profile_indexes/profiles_{name_og}.index")
    history_present = os.path.exists(archive_path+f"/histories/history_{name_og}.data")
    failed = os.path.exists(archive_path+f"/failed/run_{name_og}.log")
    if runlog_present and profiles_archived and history_present and not failed:
        print(f"Track {name_og} already done. Skipping...")
        return True
    else:
        print(f"Track {name_og} not previously done. Running...")
        return False

def evo_star_i(name, mass, metallicity, v_surf_init, param={}, index=None, archive_path = "grid_archive",
               gyre_flag=False, logging=True, parallel=False, cpu_this_process=1, produce_track=True, 
               uniform_rotation=True, additional_params={}, trace=None, overwrite=False):
    """
    dSct star evolution. Testing function. i'th track.

    Parameters
    ----------
    name : str
        Name/Dir to be created by MESA-PORT for the run. Used for archiving. 
        If two runs have the same name, the second run will overwrite the first.
    mass : float
        Initial mass in solar masses
    metallicity : float
        Initial metallicity
    v_surf_init : float
        Initial surface rotation velocity in km/s
    param : dict, optional
        Additional parameters to be set in inlist, by default {}
    index : int, optional
        Index of this track. 
        If index is 0, those inlist files will be saved in the archive 
        directory for future reference. Index is None by default.
    archive_path : str, optional
        Path to archive directory, by default "grid_archive"
    gyre_flag : bool, optional
        Run GYRE, by default False
    logging : bool, optional
        MESA-PORT logging, by default True
    parallel : bool, optional
        MESA-PORT parallel, by default False
    cpu_this_process : int, optional
        Number of cores to use per process, by default 1
    produce_track : bool, optional
        To produce the track, by default True. 
        Useful if you want to run GYRE on an existing track.
    uniform_rotation : bool, optional
        Uniform rotation, by default True
    additional_params : dict, optional
        Additional parameters to be set in inlist, by default {}
    trace : dict, optional
        MESA-PORT trace, by default None
    overwrite : bool, optional
        Overwrite existing directories, by default False
    """
    print('Start time: ', time.strftime("%H:%M:%S", time.localtime()))
    name_og = ''.join(name)
    archive_path = os.path.abspath(archive_path)
    os.environ["OMP_NUM_THREADS"] = str(cpu_this_process)
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    print(f"Mass: {mass} MSun, Z: {metallicity}, v_init: {v_surf_init} km/s")
    if param is not None:
        print(f"Primary test parameters: {param}")
    if additional_params is not None:
        print(f"Additional parameters: {additional_params}")

    HOME = os.environ["HOME"]
    if not os.path.exists(archive_path):
        helper.create_grid_dirs(overwrite=overwrite, archive_path=archive_path)
    try:
        jobfs = os.path.join(os.environ["PBS_JOBFS"], "gridwork")
        name = os.path.abspath(os.path.join(jobfs, name_og))
    except KeyError:
        jobfs = os.path.abspath("./gridwork")
        name = os.path.join(jobfs, name_og)

    ## Create working directory
    proj = ProjectOps(name)   
    initial_mass = mass
    Zinit = metallicity

    failed = True   ## Flag to check if the run failed
    previously_done = check_if_done(name_og, archive_path)
    if produce_track and not previously_done:
        if not os.path.exists(f"{archive_path}/inlists/inlists_{name_og}"):
            os.mkdir(f"{archive_path}/inlists/inlists_{name_og}")
        start_time = time.time()
        proj.create(overwrite=True) 
        with open(f"{name}/run.log", "a+") as f:
            f.write(f"Mass: {mass} MSun, Z: {metallicity}, v_init: {v_surf_init} km/s\n")
            f.write(f"CPU: {cpu_this_process}\n\n")
        star = MesaAccess(name)
        rotation_init_params = {'change_rotation_flag': True,   ## False for rotation off until near zams
                                'new_rotation_flag': True,
                                'change_initial_rotation_flag': True,
                                'set_initial_surface_rotation_v': True,
                                'set_surface_rotation_v': True,
                                'new_surface_rotation_v': v_surf_init,
                                'relax_surface_rotation_v' : True,
                                'num_steps_to_relax_rotation' : 100,  ## Default value is 100
                                'relax_omega_max_yrs_dt' : 1.0E-5}   ## Default value is 1.0E9
        
        # convergence_helper = {"restore_mesh_on_retry" : True} 
        # convergence_helper = {"convergence_ignore_equL_residuals" : True}        
        # convergence_helper = {"steps_before_use_gold_tolerances" : 100, "use_gold_tolerances" : False}  
        # convergence_helper = {'Pextra_factor' : 2, "steps_before_use_gold_tolerances" : 100, 
        #                       "use_gold_tolerances" : False, 'scale_max_correction' : 0.1}
        convergence_helpers = [{"restore_mesh_on_retry" : True}, {"steps_before_use_gold_tolerances" : 100, 
                                                                  "use_gold_tolerances" : False},
                                {'Pextra_factor' : 2, "steps_before_use_gold_tolerances" : 100, 
                                 "use_gold_tolerances" : False, 'scale_max_correction' : 0.1},
                                 {"convergence_ignore_equL_residuals" : True}]
        
        
        retry = 0
        total_retries = 4
        retry_type, terminate_type = None, None
        failed_phase = None
        while retry<=total_retries and failed:
            template_path = "./src/templates_dev"
            # template_path = "./src/templates"
            inlist_file = f"{template_path}/inlist_template"
            star.load_HistoryColumns(f"{template_path}/history_columns.list")
            star.load_ProfileColumns(f"{template_path}/profile_columns.list")
            star.load_Extras(f"{template_path}/run_star_extras_Dziembowski2.f")
            stopping_conditions = [{"stop_at_phase_PreMS":True}, {"stop_at_phase_ZAMS":True}, {"max_age":5E+007}, {"stop_at_phase_TAMS":True}, "ERGB"]
            # max_timestep = [1e4, 1e5, 1e5, 2e6, 1E7]    ## For GRID
            # profile_interval = [1, 1, 1, 5, 5]
            max_timestep = [1e4, 1e6, 1e6, 2e6, 1E7]    ## For tests
            profile_interval = [5, 5, 5, 5, 5]
            phases_params = helper.phases_params(initial_mass, Zinit)     
            phases_names = list(phases_params.keys())
            if failed_phase is None or failed_phase == "Create Pre-MS Model":
                proj.clean()
                proj.make(silent=True)
            else:
                print("Retrying from failed phase: ", failed_phase)
                phase_idx = phases_names.index(failed_phase)
                phases_names = phases_names[phase_idx:]
                stopping_conditions = stopping_conditions[phase_idx:]
                max_timestep = max_timestep[phase_idx:]
                profile_interval = profile_interval[phase_idx:]
            ## Loop over phases, set parameters and run
            for phase_name in phases_names:
                try:
                    ## Run from inlist template by setting parameters for each phase
                    print(phase_name)
                    star.load_InlistProject(inlist_file)
                    star.set(phases_params[phase_name], force=True)

                    ## ADDITIONAL PARAMETERS
                    if len(param) > 0 and param is not None:
                        star.set(param, force=True)
                    if len(additional_params) > 0 and additional_params is not None:
                        star.set(additional_params, force=True)

                    ## History and profile interval
                    star.set({'history_interval':1, "profile_interval":profile_interval.pop(0), "max_num_profile_models":6000})
                    
                    #Timestep control
                    star.set({"max_years_for_timestep": max_timestep.pop(0)}, force=True)
                    
                    ## Stopping conditions
                    stopping_condition = stopping_conditions.pop(0)
                    if  stopping_condition == "ERGB":
                        ergb_params = {'Teff_lower_limit' : 6000}
                        star.set(ergb_params, force=True)
                    else:
                        star.set(stopping_condition, force=True)
                        
                    ## Retries
                    if retry > 0:
                        if "delta_lgTeff" in retry_type:
                            teff_helper(star, retry)
                    if retry > 0 and "residual" in retry_type and phase_name == failed_phase:
                        if len(convergence_helpers[retry-1]) > 0:
                            star.set(convergence_helpers[retry-1], force=True)

                    #### RUN ####
                    ## proj.run() for first run, proj.resume() for subsequent runs
                    ## These raise exceptions if the run fails and return termination code + age otherwise
                    if phase_name == "Create Pre-MS Model":
                        ## Save a copy of the inlist for reference. Needs to be done here so that phase information is retained
                        shutil.copy(f"{name}/inlist_project", archive_path+f"/inlists/inlists_{name_og}/inlist_{phase_name.replace(' ', '_')}")
                        ## Initial/Pre-MS run
                        termination_code, age = proj.run(logging=logging, parallel=parallel, trace=trace, env=os.environ.copy())
                        print(f"End age: {age:.2e} yrs")
                        print(f"Termination code: {termination_code}\n")
                    elif phase_name == "Pre-MS Evolution":
                        ## Initiate rotation
                        if v_surf_init>0:
                            star.set(rotation_init_params, force=True)
                            ## Rotation type: uniform or differential
                            star.set({"set_uniform_am_nu_non_rot": uniform_rotation}, force=True)
                        ## Save a copy of the inlist for reference. Needs to be done here so that phase information is retained
                        shutil.copy(f"{name}/inlist_project", archive_path+f"/inlists/inlists_{name_og}/inlist_{phase_name.replace(' ', '_')}")
                        ## Resume
                        termination_code, age = proj.resume(logging=logging, parallel=parallel, trace=trace, env=os.environ.copy())
                        print(f"End age: {age:.2e} yrs")
                        print(f"Termination code: {termination_code}\n")
                    elif phase_name == "Early MS Evolution":
                        ## Save a copy of the inlist for reference. Needs to be done here so that phase information is retained
                        shutil.copy(f"{name}/inlist_project", archive_path+f"/inlists/inlists_{name_og}/inlist_{phase_name.replace(' ', '_')}")
                        ## Resume 
                        termination_code, age = proj.resume(logging=logging, parallel=parallel, trace=trace, env=os.environ.copy())
                        print(f"End age: {age:.2e} yrs")
                        print(f"Termination code: {termination_code}\n")
                        failed = False
                        break
                except Exception as e:
                    failed = True
                    print(e)
                    retry_type, terminate_type = helper.read_error(name)
                    break
                except KeyboardInterrupt:
                    raise KeyboardInterrupt
                else:
                    failed = False
            if failed:
                retry += 1
                with open(f"{name}/run.log", "a+") as f:
                    retry_from_phase_idx = phases_names.index(phase_name)-1 if phases_names.index(phase_name)>0 else 0
                    if "photo does not exist" in terminate_type:
                        f.write(f"\"Photo does not exist. Retrying previous phase\"\n")
                        failed_phase = phases_names[retry_from_phase_idx]
                    elif "ERROR" in terminate_type:
                        f.write(f"\"ERROR\" encountered\n")
                        break
                    else:
                        failed_phase = phases_names[retry_from_phase_idx]
                        f.write(f"\nMass: {mass} MSun, Z: {metallicity}, v_init: {v_surf_init} km/s\n")
                        f.write(f"Failed at phase: {phase_name}\n")
                        f.write(f"Terminate type: {terminate_type}\n")
                        f.write(f"Retry type: {retry_type}\n")
                        f.write(f"Retrying from phase: {failed_phase}\n")
                        if "delta_lgTeff" in retry_type:
                            f.write(f"Retrying with \"T_eff helper\"\n")
                        else:
                            f.write(f"Retrying with \"convergence helper\": {convergence_helpers[0]}\n")
                    if retry == total_retries:
                        f.write(f"Max retries reached. Model skipped!\n")
                        break
                    
        end_time = time.time()
        with open(f"{name}/run.log", "a+") as f:
            f.write(f"Total time: {end_time-start_time} s\n\n")
        if failed:
            shutil.copy(f"{name}/run.log", archive_path+f"/failed/run_{name_og}.log")
    else:
        failed = False

    if not failed and not previously_done:
        if gyre_flag:   
            try:
                res = run_gyre(proj, name, Zinit, cpu_this_process=cpu_this_process)
                res = True if res == 0 else False
            except Exception as e:
                res = False
                print("Gyre failed for track ", name_og)
                print(e)
            shutil.copy(f"{name}/gyre.log", archive_path+f"/gyre/gyre_{name_og}.log")
        shutil.copy(f"{name}/run.log", archive_path+f"/runlogs/run_{name_og}.log")
        
        archiving_successful = False
        try:
            print("Archiving LOGS...")
            helper.archive_LOGS(name, name_og, True, (gyre_flag and res), archive_path, remove=False)
            archive_successful = True
        except Exception as e:
            print("Archiving failed for track ", name_og)
            print(e)
            archiving_successful = False
        with open(archive_path+f"/runlogs/run_{name_og}.log", "a+") as f:
            if archiving_successful:
                f.write(f"LOGS archived!\n")
            else:
                f.write(f"LOGS archiving failed!\n")

def run_gyre(proj, name, Z, cpu_this_process=1):
    start_time = time.time()
    print("[bold green]Running GYRE...[/bold green]")
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
    os.environ['OMP_NUM_THREADS'] = '1'
    file_format = "GSM"
    profiles, gyre_input_params = gyre.get_gyre_params(name, Z, file_format=file_format, run_on_cool=True)
    if len(profiles) == 0:
        file_format = "GYRE"
        profiles, gyre_input_params = gyre.get_gyre_params(name, Z, file_format=file_format, run_on_cool=True)
    if len(profiles) > 0:
        profiles = [profile.split('/')[-1] for profile in profiles]
        res = proj.runGyre(gyre_in=os.path.expanduser("./src/templates/gyre_rot_template_ell3.in"), 
                     files=profiles, data_format=file_format, profiles_dir="LOGS",
                    logging=True, parallel=True, n_cores=cpu_this_process, gyre_input_params=gyre_input_params)
        with open(f"{name}/run.log", "a+") as f:
            if res == True:
                f.write(f"GYRE run complete!\n")
            else:
                f.write(f"GYRE failed!\n")
    else:
        res = False
        with open(f"{name}/run.log", "a+") as f:
            f.write(f"GYRE skipped: no profiles found, possibly because all models had T_eff < 6000 K\n")
    end_time = time.time()
    with open(f"{name}/gyre.log", "a+") as f:
        f.write(f"Total time: {end_time-start_time} s\n\n")
    return res



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run parameter tests for MESA",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--index", type=int, default=0, 
                        help="Index of the parameter set to run")
    parser.add_argument("-c", "--cores", type=int, default=1, 
                        help="Number of cores to use per process")
    parser.add_argument("-o", "--overwrite", action="store_true",
                        help="Overwrite existing directories")
    parser.add_argument("-d", "--directory", type=str, default="grid_archive",
                        help="Name of the directory to save outputs")
    args = parser.parse_args()
    config = vars(args)
    index = config["index"]
    cpu_per_process = config["cores"]
    overwrite = config["overwrite"]
    archive_dir = config["directory"]

    if index == 0 and overwrite:
        overwrite = True
    else:
        overwrite = False

    rewrite_input_file = True
    if index == 0 and rewrite_input_file:
        M_sample = [1.96, 1.98, 2.0, 2.2]
        # M_sample = [1.2, 1.4, 1.6, 1.8, 2.0, 2.2]
        # M_sample = [1.52]
        # Z_sample = [0.008]
        Z_sample = [0.014, 0.015, 0.016, 0.017, 0.018]
        # Z_sample = [0.002, 0.006, 0.01, 0.014, 0.018, 0.022, 0.026]
        V_sample = [8]
        # param_samples = [{'overshoot_f(1)':0.045, 'overshoot_f0(1)':0.005},
        #                     {'overshoot_f(1)':0.055, 'overshoot_f0(1)':0.005},
        #                     {'overshoot_f(1)':0.065, 'overshoot_f0(1)':0.005},
        #                     {'overshoot_f(1)':0.075, 'overshoot_f0(1)':0.005},
        #                     {'overshoot_f(1)':0.085, 'overshoot_f0(1)':0.005}]
        param_samples = [{'overshoot_f(1)':0.017, 'overshoot_f0(1)':0.002},
                            {'overshoot_f(1)':0.022, 'overshoot_f0(1)':0.002},
                            {'overshoot_f(1)':0.027, 'overshoot_f0(1)':0.002},
                            {'overshoot_f(1)':0.032, 'overshoot_f0(1)':0.002}]
        # param_samples = param_samples.ov_samples
        combinations = list(product(M_sample, Z_sample, V_sample, param_samples))
        # print("Number of combinations: ", len(combinations))
        # exit()
        

        M = []
        Z = []
        V = []
        params = []
        names = []
        for i, (m, z, v, param) in enumerate(combinations):
            M.append(m)
            Z.append(z)
            V.append(v)
            if param is not None:
                params.append(param)
                names.append(f"m{m}_z{z}_v{v}_param{param_samples.index(param)}")
            else:
                names.append(f"m{m}_z{z}_v{v}")

        df = pd.DataFrame({"track": names, "M": M, "Z": Z, "V": V, "param": params}) if len(params)>0 else pd.DataFrame({"track": names, "M": M, "Z": Z, "V": V})
        df.to_csv('track_inputs.csv', index=False)
    df = pd.read_csv('track_inputs.csv')
    name = df["track"].loc[index]
    M = float(df["M"].loc[index])
    Z = float(df["Z"].loc[index])
    V = float(df["V"].loc[index])
    param = eval(df["param"].loc[index]) if 'param' in df.columns else {}

    additional_params = {}
    # additional_params = {}

    trace = ['surf_avg_v_rot', 'surf_avg_omega_div_omega_crit', 
            'log_total_angular_momentum',
            'surf_escape_v', 'log_g', 'log_R', 'star_mass']

    evo_star_i(name=name, mass=M, metallicity=Z, v_surf_init=V, param=param, 
               index=index, gyre_flag=False, logging=True, overwrite=overwrite,
               parallel=False, cpu_this_process=cpu_per_process, trace=trace,
               produce_track=True, uniform_rotation=True, additional_params=additional_params, archive_path=archive_dir)
    
    if index == 0:
        shutil.copy("track_inputs.csv", archive_dir)

    print("Done!")


