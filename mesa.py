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

