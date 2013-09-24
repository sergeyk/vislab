import pandas as pd
import os
import numpy as np
import time
import pymongo
import rq
import redis
import sys
import shlex
import socket
import tempfile
import cPickle
import subprocess


def exclude_ids_in_collection(image_ids, collection):
    """
    Exclude ids already stored in the collection.
    Useful for submitting map jobs.
    """
    computed_image_ids = [
       x['image_id'] for x in collection.find(fields=['image_id'])
    ]
    num_ids = len(image_ids)
    image_ids = list(set(image_ids) - set(computed_image_ids))
    print("Cut down on {} existing out of {} total image ids.".format(
        num_ids - len(image_ids), num_ids))
    return image_ids


def load_or_generate_df(filename, generator_fn, force):
    if not force and os.path.exists(filename):
        df = pd.read_hdf(filename, 'df')
    else:
        df = generator_fn()
        df.to_hdf(filename, 'df', mode='w')
    return df


def running_on_icsi():
    """
    Return True if this script is running on the ICSI cluster.
    """
    return socket.gethostname().endswith('ICSI.Berkeley.EDU')


def get_mongodb_client():
    """
    Establish connection to MongoDB.
    """
    try:
        host = 'flapjack' if running_on_icsi() else 'localhost'
        connection = pymongo.MongoClient(host, 27666)
        return connection
    except pymongo.errors.ConnectionFailure:
        raise Exception(
            "Need a MongoDB server running on {}, port 27666".format(host))


def get_redis_conn():
    host = 'flapjack' if running_on_icsi() else 'localhost'
    redis_conn = redis.Redis(host)
    return redis_conn


def chunk(function, args_list):
    return [function(*args) for args in args_list]


def map_through_rq(
        function, args_list, name='default', aggregate=False,
        num_workers=1, chunk_size=1, cpus_per_task=1, mem=3000,
        max_time='12:0:0', result_ttl=640, timeout=320, async=True):
    """
    Map function onto args_list by submitting the jobs to a Redis queue,
    and launching background workers to process them.
    Return when all jobs are done.

    This function does not return anything.
    The jobs are expected to write to files or MongoDB for later collection.

    Parameters
    ----------
    function: function
        Function that will be called.
    args_list: list
        List of tuples that will be mapped with the function.
    name: string ['default']
        Name will be used for the Redis queue.
    aggregate: bool [False]
        If True, return the results of all jobs.
        NOTE: results are cached for only 12 hours.
    num_workers: int [1]
        Number of rq-worker instances to launch.
    chunk_size: int [1]
        Must be a non-negative int. If chunk_size > 1, jobs are processed
        by workers in chunks of the given size instead of one by one.
    cpus_per_task: int [1]
        Number of CPUs that each worker will use.
    mem: int [1500]
        Maximum amount of memory that each worker will use, in MB.
    max_time: string ['2:0:0']
        Maximum amount of time a worker may live, as 'hours:minutes:seconds'.
    result_ttl: int [43200]
        Time that the result of a job lives in Redis for reading, in seconds.
    timeout: int [320]
        Time that the worker has to respond once it starts a job, in seconds.
    async: boolean [True]
        If False, will run job synchronously in same thread---useful to debug.
    """
    assert(chunk_size > 0)

    # Establish connection to Redis queue.
    redis_conn = get_redis_conn()
    fq = rq.Queue('failed', connection=redis_conn)
    q = rq.Queue(name, connection=redis_conn, async=async)

    # Check the failed queue for old jobs and get rid of them.
    failed_jobs = [j.cancel() for j in fq.get_jobs() if j.origin == name]
    print("Canceled {} failed jobs are left over from previous run.".format(len(failed_jobs)))

    # Empty the queue and fill it with jobs (result_ttl is the caching time of results, in seconds).
    print("Queing jobs...")
    q.empty()
    t = time.time()
    if chunk_size > 1:
        chunked_args_list = [(function, args_list[i:i+chunk_size]) for i in range(0, len(args_list), chunk_size)]
        jobs = [q.enqueue_call(func=chunk, args=chunked_args, timeout=timeout, result_ttl=result_ttl) for chunked_args in chunked_args_list]
    else:
        jobs = [q.enqueue_call(func=function, args=args, timeout=timeout, result_ttl=result_ttl) for args in args_list]
    print("...finished in {:.3f} s".format(time.time() - t))

    if async:
        # Start worker(s) set to die when the queue runs out.
        # If running on ICSI, launch workers through SLURM.
        # Otherwise, launch in background.
        print("Starting {} workers...".format(num_workers))
        cmd = "rqworker --burst {}".format(name)
        if running_on_icsi():
            redis_hostname = 'flapjack'
            job_log_dirname = makedirs('data_shared/rqworkers')
            cmd = "srun -p vision --cpus-per-task={} --mem={} --time={} --output={}/{}_%j-out.txt rqworker --host {} --burst {}".format(
                cpus_per_task, mem, max_time, job_log_dirname, name, redis_hostname, name)
        print(cmd)
        pids = []
        for i in range(num_workers):
            time.sleep(np.random.rand())  # stagger the jobs a little bit
            pids.append(subprocess.Popen(shlex.split(cmd),
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE))

        # Wait until all jobs are completed.
        t = time.time()
        known_jobs = {}
        while True:
            for job in jobs:
                if job not in known_jobs:
                    if job.is_finished:
                        known_jobs[job] = 0
                    elif job.is_failed:
                        known_jobs[job] = 1
            num_failed = sum(known_jobs.values())
            num_succeeded = len(known_jobs) - num_failed
            sys.stdout.write('\r{:.1f} s passed, {} succeeded / {} failed out of {} total'.format(time.time() - t, num_succeeded, num_failed, len(jobs)))
            sys.stdout.flush()
            if num_succeeded + num_failed == len(jobs):
                break
            time.sleep(2)
        sys.stdout.write('\n')
        sys.stdout.flush()
    print('Done with all jobs.')

    # Print some statistics about the run.
    failed_jobs = [j for j in fq.get_jobs() if j.origin == name]
    print("{} jobs failed and went into the failed queue.".format(len(failed_jobs)))

    # If requested, aggregate and return the results.
    if aggregate:
        results = [job.result for job in jobs]
        if chunk_size > 1:
            # Watch out for some jobs retuning None results.
            all_results = []
            for result in results:
                if result is not None:
                    all_results += result
            results = all_results
        return results

    return None


def pickle_function_call(func_name, args):
    f, temp_filename = tempfile.mkstemp()
    with open(temp_filename, 'w') as f:
        cPickle.dump((func_name, args), f)
    c = "import os; import cPickle; f = open('{0}'); func, args = cPickle.load(f); f.close(); os.remove('{0}'); func(*args)".format(temp_filename)
    return c


def run_through_bash_script(cmds, filename=None, verbose=False):
    """
    Write out given commands to a bash script file and execute it.
    This is useful when the commands to run include pipes, or are chained.
    subprocess is not too easy to use in those cases.

    Parameters
    ----------
    cmds: list of string
    filename: string or None [None]
        If None, a temporary file is used and deleted after.
    verbose: bool [False]
        If True, output the commands that will be run.
    """
    if verbose:
        print("Commands that will be run:")
        for cmd in cmds:
            print cmd

    remove_file = False
    if filename is None:
        f, filename = tempfile.mkstemp()
        remove_file = True

    with open(filename, 'w') as f:
        for cmd in cmds:
            f.write(cmd + '\n')

    p = subprocess.Popen(['bash', filename])
    out, err = p.communicate()

    if remove_file:
        os.remove(filename)
    if not p.returncode == 0:
        print(out)
        print(err)
        raise Exception("Script exited with code {}".format(p.returncode))


def makedirs(dirname):
    if os.path.exists(dirname):
        return dirname
    try:
        os.makedirs(dirname)
    except OSError:
        pass
    except:
        raise
    return dirname
