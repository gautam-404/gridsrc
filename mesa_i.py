import os, subprocess
from mesaport import MesaAccess, ProjectOps
import argparse
from itertools import product
import time
import pandas as pd
import shutil
import platform

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

def evo_star_i(name, mass, metallicity, v_surf_init, param={}, archive_path="grid_archive",
               logging=True, parallel=False, cpu_this_process=1, produce_track=True, 
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
    archive_path : str, optional
        Path to archive directory, by default "grid_archive"
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
    print('Start Date: ', time.strftime("%d-%m-%Y", time.localtime()))
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

    if "macOS" in platform.platform():
        grid_name = archive_path.split('/')[-1].split('grid_archive_')[-1]
        jobfs = os.path.abspath(f"./gridwork_{grid_name}")
        name = os.path.join(jobfs, name_og)
    else:
        try:
            jobfs = os.path.join(os.environ["PBS_JOBFS"], "gridwork")
            name = os.path.abspath(os.path.join(jobfs, name_og))
        except KeyError:
            jobfs = os.path.join(os.environ["TMPDIR"], "gridwork")
            name = os.path.abspath(os.path.join(jobfs, name_og))
        else:
            grid_name = archive_path.split('/')[-1].split('grid_archive_')[-1]
            jobfs = os.path.abspath(f"./gridwork_{grid_name}")
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
            # template_path = "./src/templates_dev"
            template_path = "./src/templates"
            inlist_file = f"{template_path}/inlist_template"
            star.load_HistoryColumns(f"{template_path}/history_columns.list")
            star.load_ProfileColumns(f"{template_path}/profile_columns.list")
            stopping_conditions = [{"stop_at_phase_PreMS":True}, {"stop_at_phase_ZAMS":True}, {"max_age":50e6}, {"stop_at_phase_TAMS":True}, "ERGB"]
            # max_timestep = [1e4, 1e5, 1e5, 2e6, 1E7]    ## For GRID
            # profile_interval = [1, 1, 1, 5, 5]
            max_timestep = [1e4, 1e6, 1e6, 2e6, 1E7]    ## For tests
            profile_interval = [1, 3, 3, 5, 5]
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
                    # stopping_condition = stopping_conditions.pop(0)
                    stopping_condition = stopping_conditions[phases_names.index(phase_name)]
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

                    ## Rotation type: uniform or differential
                    star.set({"set_uniform_am_nu_non_rot": uniform_rotation}, force=True)

                    #### RUN ####
                    ## proj.run() for first run, proj.resume() for subsequent runs
                    ## These raise exceptions if the run fails and return termination code + age otherwise
                    stopping_conditions = [{"stop_at_phase_PreMS":True}, {"stop_at_phase_ZAMS":True}, {"max_age":50e6}, {"stop_at_phase_TAMS":True}, "ERGB"]
                    reqd_phases = ["Create Pre-MS Model",  "Pre-MS Evolution", "Early MS Evolution", "Evolution to TAMS", "Evolution post-MS"]
                    shutil.copy(f"{name}/inlist_project", archive_path+f"/inlists/inlists_{name_og}/inlist_{phase_name.replace(' ', '_')}")
                    if phase_name == reqd_phases[0]:
                        termination_code, age = proj.run(logging=logging, parallel=parallel, trace=trace, env=os.environ.copy())
                    elif phase_name == reqd_phases[1]:
                        ## Initiate rotation
                        if v_surf_init>0:
                            star.set(rotation_init_params, force=True)
                        termination_code, age = proj.resume(logging=logging, parallel=parallel, trace=trace, env=os.environ.copy())
                    elif phase_name in reqd_phases[2:]:
                        termination_code, age = proj.resume(logging=logging, parallel=parallel, trace=trace, env=os.environ.copy())
                    print(f"End age: {age:.2e} yrs")
                    print(f"Termination code: {termination_code}\n")
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
        shutil.copy(f"{name}/run.log", archive_path+f"/runlogs/run_{name_og}.log")
        archiving_successful = False
        try:
            print("Archiving LOGS...")
            helper.archive_LOGS(name, name_og, True, False, archive_path)
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