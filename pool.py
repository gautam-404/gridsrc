import subprocess
import os, psutil

from multiprocessing.pool import Pool as MPool
from ray.runtime_env import RuntimeEnv
from rich import progress, print, console

from . import helper

def ray_pool(func, args, length, cpu_per_process=16, config={"cores" : None}, initializer=helper.mute):
    '''
    Multiprocessing pool with ray.

    Parameters
    ----------
    func : function
        Function to be run in parallel.
    args : list
        List of arguments to be passed to func.
    length : int
        Length of args.
    cpu_per_process : int, optional
        Number of CPUs per process. The default is 16.
    config : dict, optional
        Dictionary of configuration parameters. The default is {"cores" : None}.
    '''
    import ray  
    from ray.util.multiprocessing import Pool as RayPool 
    try:
        ray.init(address="auto")
    except:
        ## Start the ray cluster
        with console.Console().status("[b i][blue]Starting ray cluster...[/blue]") as status:
            start_ray()
        print("[b i][green]Ray cluster started.[/green]\n")
        ray.init(address="auto")
    print("\n[b i][blue]Ray cluster resources:[/blue]")
    print("CPUs: ", ray.cluster_resources()["CPU"])

    if config["cores"] is None:
        processors = int(ray.cluster_resources()["CPU"])
    else:
        processors = int(config["cores"])
    runtime_env = RuntimeEnv(env_vars={"OMP_NUM_THREADS": str(cpu_per_process), 
                                        "MKL_NUM_THREADS": str(cpu_per_process)})
    ray_remote_args = {"num_cpus": cpu_per_process, "runtime_env": runtime_env, 
                        "scheduling_strategy" : "DEFAULT", 
                        "max_restarts" : -1, "max_task_retries" : -1}
    n_processes = (processors // cpu_per_process)
    print(f"[b][blue]Running {min([n_processes, length])} parallel processes on {processors} cores.[/blue]")
    with progress.Progress(*helper.progress_columns()) as progressbar:
        task = progressbar.add_task("[b i green]Running...", total=length)
        with RayPool(ray_address="auto", processes=n_processes, initializer=initializer, ray_remote_args=ray_remote_args) as pool:
            for i, res in enumerate(pool.imap_unordered(func, args)):
                progressbar.advance(task)

def mp_pool(func, args, length, cpu_per_process=16, config={"cores" : None}, initializer=helper.mute):
    '''
    Multiprocessing pool mith python multiprocessing.

    Parameters
    ----------
    func : function
        Function to be run in parallel.
    args : list
        List of arguments to be passed to func.
    length : int
        Length of args.
    cpu_per_process : int, optional
        Number of CPUs per process. The default is 16.
    config : dict, optional
        Dictionary of configuration parameters. The default is {"cores" : None}.
    '''
    if config["cores"] is None:
        processors = int(psutil.cpu_count(logical=False))
    else:
        processors = int(config["cores"])
    os.environ["OMP_NUM_THREADS"] = str(cpu_per_process)
    n_processes = (processors // cpu_per_process)
    print(f"[b][blue]Running {min([n_processes, length])} parallel processes on {processors} cores.[/blue]")
    with progress.Progress(*helper.progress_columns()) as progressbar:
        task = progressbar.add_task("[b i green]Running...", total=length)
        with MPool(processes=n_processes, initializer=initializer) as pool:
            for i, res in enumerate(pool.imap_unordered(func, args)):
            # for i, res in enumerate(pool.istarmap(func, args)):
                progressbar.advance(task)

def stop_ray():
    '''
    Stop ray cluster.
    '''
    subprocess.call("ray stop --force".split(" "))
    subprocess.call("killall -9 pbs_tmrsh".split(" "))

def start_ray():
    '''
    Start ray cluster.
    '''
    ## this shell script stops all ray processing before starting new cluster
    subprocess.call("./src/rayCluster/ray-cluster.sh", stdout=subprocess.DEVNULL)
