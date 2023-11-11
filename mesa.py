import numpy as np
import os
import shutil
import time
import tarfile

from mesaport import ProjectOps, MesaAccess
from rich import print

from . import helper, gyre

def is_done(track_index, archive_path, save_track_path):
    '''
    Check if the model has already been calculated

    Parameters
    ----------
    track_index : int
        Track index
    save_track_path : str
        Path to save the track
    '''
    mesa_complete = False
    gyre_complete = False
    logs_archived = False
    try:
        with open(f"{archive_path}/runlogs/run_{track_index}.log", "r") as f:
            for line in f:
                if "Total time" in line:
                    mesa_complete = True
                if "GYRE run complete!" in line:
                    gyre_complete = True
                if "LOGS archived!" in line:
                    logs_archived = True
        # if os.path.exists(f"{save_track_path}/track_{track_index}.tar.gz") and logs_archived:
        if archive_path+f"/profiles/profiles_{track_index}.tar.gz" and logs_archived:
            logs_archived = True
        else:
            logs_archived = False
    except FileNotFoundError:
        print("Track not previously calculated. Calculating now...")
        return False, False, False
    if mesa_complete and gyre_complete and logs_archived:
        print("Track already calculated. Skipped!")
    return mesa_complete, gyre_complete, logs_archived


def teff_helper(star, retry):
    '''
    Increase delta_lgTeff_limit and delta_lgTeff_hard_limit for each retry if the model fails to converge with a teff error

    Parameters
    ----------
    star : MesaAccess
        MESA star object
    retry : int
        n-th retry
    '''
    delta_lgTeff_limit = star.get("delta_lgTeff_limit")
    delta_lgTeff_hard_limit = star.get("delta_lgTeff_hard_limit")
    # delta_lgTeff_limit += delta_lgTeff_limit/10
    delta_lgTeff_hard_limit += retry*delta_lgTeff_hard_limit
    star.set({"delta_lgTeff_limit": delta_lgTeff_limit, "delta_lgTeff_hard_limit": delta_lgTeff_hard_limit}, force=True)



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

    print(f"Mass: {mass} MSun, Z: {metallicity}, v_init: {v_surf_init} km/s")
    if param is not None:
        print(f"Primary test parameters: {param}")
    if additional_params is not None:
        print(f"Additional parameters: {additional_params}")

    HOME = os.environ["HOME"]
    helper.create_grid_dirs(overwrite=overwrite, archive_path=archive_path)
    os.mkdir(f"{archive_path}/inlists/inlists_{name_og}")
    try:
        jobfs = os.path.join(os.environ["PBS_JOBFS"], "gridwork")
        name = os.path.abspath(os.path.join(jobfs, name_og))
    except KeyError:
        jobfs = "./gridwork"
        name = os.path.abspath(os.path.join(jobfs, name_og))

    ## Create working directory
    proj = ProjectOps(name)   
    initial_mass = mass
    Zinit = metallicity

    failed = True   ## Flag to check if the run failed
    if produce_track:  
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
            # stopping_conditions = [{"stop_at_phase_PreMS":True}, {"stop_at_phase_ZAMS":True}, {"max_age":4e7}, {"stop_at_phase_TAMS":True}, "ERGB"]
            stopping_conditions = [{"stop_at_phase_PreMS":True}, {"stop_at_phase_ZAMS":True}, {"max_age":4e7}, {"stop_at_phase_TAMS":True}, "ERGB"]
            # max_timestep = [1e4, 1e5, 1e5, 2e6, 1E7]    ## For GRID
            # profile_interval = [1, 1, 1, 5, 5]
            max_timestep = [1e4, 1e6, 1e6, 2e6, 1E7]    ## For tests
            profile_interval = [1, 5, 5, 5, 5]
            phases_params = helper.phases_params(initial_mass, Zinit)     
            phases_names = list(phases_params.keys())
            if failed_phase is not None:
                print("Retrying from failed phase: ", failed_phase)
                phase_idx = phases_names.index(failed_phase)
                phases_names = phases_names[phase_idx:]
                stopping_conditions = stopping_conditions[phase_idx:]
                max_timestep = max_timestep[phase_idx:]
                profile_interval = profile_interval[phase_idx:]
            else:
                proj.clean()
                proj.make(silent=True)
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
                    star.set({'history_interval':1, "profile_interval":profile_interval.pop(0), "max_num_profile_models":3000})
                    
                    #Timestep control
                    star.set({"max_years_for_timestep": max_timestep.pop(0)}, force=True)
                    
                    ## Stopping conditions
                    stopping_condition = stopping_conditions.pop(0)
                    if  stopping_condition == "ERGB":
                        ergb_params = {'Teff_lower_limit' : 6000}
                        star.set(ergb_params, force=True)
                    else:
                        star.set(stopping_condition, force=True)

                    ### Checks
                    ## Rotation type: uniform or differential
                    star.set({"set_uniform_am_nu_non_rot": uniform_rotation}, force=True)

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
                    if "photo does not exist" in terminate_type:
                        f.write(f"\"Photo does not exist. Exiting.\"\n")
                        break
                    elif "ERROR" in terminate_type:
                        f.write(f"\"ERROR\" encountered\n")
                        break
                    if retry == total_retries:
                        f.write(f"Max retries reached. Model skipped!\n")
                        break
                    f.write(f"\nMass: {mass} MSun, Z: {metallicity}, v_init: {v_surf_init} km/s\n")
                    f.write(f"Failed at phase: {phase_name}\n")
                    failed_phase = phase_name
                    f.write(f"Terminate type: {terminate_type}\n")
                    f.write(f"Retry type: {retry_type}\n")
                    if "delta_lgTeff" in retry_type:
                        f.write(f"Retrying with \"T_eff helper\"\n")
                    else:
                        f.write(f"Retrying with \"convergence helper\": {convergence_helpers[0]}\n")
        end_time = time.time()
        with open(f"{name}/run.log", "a+") as f:
            f.write(f"Total time: {end_time-start_time} s\n\n")
        if failed:
            shutil.copy(f"{name}/run.log", archive_path+f"/failed/run_{name_og}.log")
    else:
        failed = False

    if not failed:
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
            helper.archive_LOGS(name, name_og, True, (gyre_flag and res), archive_path)
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



# def evo_star(args, **kwargs):
def evo_star(args):
    '''
    Run MESA evolution for a single star
    
    Parameters
    ----------

    args : tuple
        (M, Z, V_surf_in, track_index)
    kwargs : dict
        gyre_flag : bool
            Run GYRE on the model
        save_track : bool
            Save the track. Default is True
        logging : bool
            Whether to log the evolution in a run.log file. Default is True
        parallel : bool
            Whether this function is called in parallel. Default is True
        cpu_this_process : int
            Number of cores to use. Default is 12
        slice_start : int
            Start index of the track. Default is 0
        uniform_rotation : bool
            Set the rotation to uniform. Default is True
        trace : list
            Parameters to trace to be passed to MESA. Default is None
        save_track_path : str
            Path to save the track
    '''
    M, Z, V_surf_in, track_index, gyre_flag, save_track, logging,\
    parallel, cpu_this_process, slice_start, uniform_rotation, trace, save_track_path = args

    # unpack args
    # M, Z, V_surf_in, track_index = args
    # # unpack kwargs
    # gyre_flag, save_track, logging, parallel, cpu_this_process,\
    #     slice_start, uniform_rotation, trace, save_track_path = \
    #         kwargs.get('gyre_flag', True), kwargs.get('save_track', True),\
    #             kwargs.get('logging', True), kwargs.get('parallel', True),\
    #                 kwargs.get('cpu_this_process', 12), kwargs.get('slice_start', 0),\
    #                 kwargs.get('uniform_rotation', True), kwargs.get('trace', []),\
    #                     kwargs.get('save_track_path', None)
    track_index += slice_start
    print(f"Track index: {track_index}")

    HOME = os.environ["HOME"]
    archive_path = os.path.join(HOME, "workspace/GRID/grid_archive")
    try:
        jobfs = os.path.join(os.environ["PBS_JOBFS"], "gridwork")
        name = os.path.abspath(os.path.join(jobfs, f"track_{track_index}"))
        if save_track_path is not None:
            save_track_path = "/g/data/qq01/urot_tracks"
    except KeyError:
        jobfs = os.path.join(HOME, "workspace/GRID/gridwork")
        name = os.path.abspath(os.path.join(jobfs, f"track_{track_index}"))
        if save_track_path is not None:
            save_track_path = archive_path + f'/tracks/track_{track_index}'

    ### Check if already calculated
    mesa_complete, gyre_complete, logs_archived = is_done(track_index, archive_path, save_track_path)
    failed = not mesa_complete or not logs_archived

    if failed:
        start_time = time.time()
        print(f"Mass: {M} MSun, Z: {Z}, v_init: {V_surf_in} km/s")
        proj = ProjectOps(name)
        proj.create(overwrite=True)
        with open(f"{name}/run.log", "a+") as f:
            f.write(f"Mass: {M} MSun, Z: {Z}, v_init: {V_surf_in} km/s\n")
            f.write(f"CPU: {cpu_this_process}\n\n")
        star = MesaAccess(name)
        star.load_HistoryColumns("./src/templates/history_columns.list")
        star.load_ProfileColumns("./src/templates/profile_columns.list")

        rotation_init_params = {'change_rotation_flag': True,   ## False for rotation off until near zams
                                'new_rotation_flag': True,
                                'change_initial_rotation_flag': True,
                                'set_initial_surface_rotation_v': True,
                                'set_surface_rotation_v': True,
                                'new_surface_rotation_v': V_surf_in,
                                'relax_surface_rotation_v' : True,
                                'num_steps_to_relax_rotation' : 100,  ## Default value is 100
                                'relax_omega_max_yrs_dt' : 1.0E-5}   ## Default value is 1.0E9
        
        convergence_helper = {"convergence_ignore_equL_residuals" : True}  

        inlist_template = "./src/templates/inlist_template"

        retry = 0
        max_retries = 4
        retry_type, terminate_type = None, None
        while retry<=max_retries and failed:
            proj.clean()
            proj.make(silent=True)
            phases_params = helper.phases_params(M, Z)   
            phases_names = phases_params.keys()
            # stopping_conditions = [{"stop_at_phase_ZAMS":True}, {"max_age":4e7}, {"stop_at_phase_TAMS":True}, "ERGB"]
            terminal_age = np.round(2500/M**2.5,1)*1.0E6
            print(f"Terminal age: {terminal_age:.3e} yrs")
            stopping_conditions = [{"stop_at_phase_ZAMS":True}, {"max_age":4e7}, 
                            {"stop_at_phase_TAMS":True, 'max_age':terminal_age}, "ERGB"]
            max_timestep = [1E4, 1E5, 2E6, 1E7]
            profile_interval = [1, 1, 5, 5]
            for phase_name in phases_names:
                try:
                    ## Run from inlist template by setting parameters for each phase
                    star.load_InlistProject(inlist_template)
                    print(phase_name)
                    star.set(phases_params[phase_name], force=True)

                    ## History and profile interval
                    star.set({'history_interval':1, "profile_interval":profile_interval.pop(0), "max_num_profile_models":5000})
                    
                    ##Timestep 
                    star.set({"max_years_for_timestep": max_timestep.pop(0)}, force=True)
                    
                    ## Stopping conditions
                    stopping_condition = stopping_conditions.pop(0)
                    if  stopping_condition == "ERGB":
                        ergb_params = {'Teff_lower_limit' : 6000}
                        star.set(ergb_params, force=True)
                    else:
                        star.set(stopping_condition, force=True)

                    ### Checks
                    if uniform_rotation:
                        star.set({"set_uniform_am_nu_non_rot": True}, force=True)
                    if retry > 0:
                        if "delta_lgTeff" in retry_type:
                            teff_helper(star, retry)
                        else:
                            star.set(convergence_helper, force=True)

                    ## RUN
                    if phase_name == "Evolution to ZAMS":
                        ## Initiate rotation
                        if V_surf_in>0:
                            star.set(rotation_init_params, force=True)
                        print(f"End age: {proj.run(logging=logging, parallel=parallel, trace=trace):.2e} yrs\n")
                    else:
                        ########## Skip post-MS evolution for now ##########
                        if phase_name == "Evolution post-MS":
                            continue
                        ####################################################
                        print(f"End age: {proj.resume(logging=logging, parallel=parallel, trace=trace):.2e} yrs\n")
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
                    if retry == max_retries:
                        f.write(f"Max retries reached. Model skipped!\n")
                        break
                    f.write(f"\nMass: {M} MSun, Z: {Z}, v_init: {V_surf_in} km/s\n")
                    f.write(f"Failed at phase: {phase_name}\n")
                    if "delta_lgTeff" in retry_type:
                        f.write(f"Retrying with \"T_eff helper\"\n")
                    else:
                        f.write(f"Retrying with \"convergence helper\"\n")
            end_time = time.time()
            with open(f"{name}/run.log", "a+") as f:
                f.write(f"Total time: {end_time-start_time} s\n\n")
        else:
            failed = False

    if not failed:
        if gyre_flag and not gyre_complete:
            if not logs_archived:
                try:
                    start_time = time.time()
                    print("[bold green]Running GYRE...[/bold green]")
                    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
                    os.environ['OMP_NUM_THREADS'] = '1'
                    profiles, gyre_input_params = gyre.get_gyre_params(name, Z, file_format="GSM")
                    if len(profiles) > 0:
                        profiles = [profile.split('/')[-1] for profile in profiles]
                        proj.runGyre(gyre_in=os.path.expanduser("./src/templates/gyre_rot_template_ell3.in"), files=profiles, data_format="GSM", 
                                    logging=False, parallel=True, n_cores=cpu_this_process, gyre_input_params=gyre_input_params)
                        with open(f"{name}/run.log", "a+") as f:
                            f.write(f"GYRE run complete!\n")
                    else:
                        with open(f"{name}/run.log", "a+") as f:
                            f.write(f"GYRE skipped: no profiles found, possibly because all models had T_eff < 6000 K\n")
                    end_time = time.time()
                    with open(f"{name}/gyre.log", "a+") as f:
                        f.write(f"Total time: {end_time-start_time} s\n\n")
                except Exception as e:
                    print("Gyre failed for track ", name)
                    print(e)
            else:
                compressed_profiles = os.path.abspath(archive_path+f"/profiles/profiles_{track_index}.tar.gz")
                with tarfile.open(compressed_profiles, "r:gz") as tar:
                    tar.extractall(path=jobfs)
                profiles_archive = os.path.abspath(jobfs+f"profiles_{track_index}")
                profiles, gyre_input_params = gyre.get_gyre_params_archived(track_index, archive_path, profiles_archive)
                with open(f"{name}/run.log", "a+") as f.l:
                    f.write(f"GYRE run complete!\n")
        if not mesa_complete:
            ## Save runlog
            shutil.copy(f"{name}/run.log", archive_path+f"/runlogs/run_{track_index}.log")
        if not logs_archived:
            ## Archive LOGS
            helper.archive_LOGS(name, track_index, save_track, gyre_flag, archive_path, tracks_dir=save_track_path)
            with open(archive_path+f"/runlogs/run_{track_index}.log", "a+") as f:
                f.write(f"LOGS archived!\n")
    else:
        if logging:         ## If the run failed, archive the log file
            shutil.copy(f"{name}/run.log", archive_path+f"/failed/failed_{track_index}.log")
        shutil.rmtree(name)

