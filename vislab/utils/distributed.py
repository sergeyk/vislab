import rq
import time
import numpy as np
import shlex
import sys
import subprocess
import vislab
from vislab import util


def chunk(function, args_list):
    return [function(*args) for args in args_list]


def map_through_rq(
        function, args_list, name='default', aggregate=False,
        num_workers=1, chunk_size=1, cpus_per_task=1, mem=3000,
        max_time='12:0:0', result_ttl=640, timeout=640, async=True):
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
    redis_conn = util.get_redis_client()
    fq = rq.Queue('failed', connection=redis_conn)
    q = rq.Queue(name, connection=redis_conn, async=async)

    # Check the failed queue for old jobs and get rid of them.
    failed_jobs = [j.cancel() for j in fq.get_jobs() if j.origin == name]
    print("Canceled {} failed jobs are left over from previous run.".format(
        len(failed_jobs)))

    # Empty the queue and fill it with jobs (result_ttl is the caching
        # time of results, in seconds).
    print("Queing jobs...")
    q.empty()
    t = time.time()
    if chunk_size > 1:
        chunked_args_list = [
            (function, args_list[i:i + chunk_size])
            for i in range(0, len(args_list), chunk_size)
        ]
        jobs = [
            q.enqueue_call(
                func=chunk, args=chunked_args,
                timeout=timeout, result_ttl=result_ttl)
            for chunked_args in chunked_args_list
        ]
    else:
        jobs = [
            q.enqueue_call(
                func=function, args=args,
                timeout=timeout, result_ttl=result_ttl)
            for args in args_list
        ]
    print("...finished in {:.3f} s".format(time.time() - t))

    if async:
        # Start worker(s) set to die when the queue runs out.
        # If running on ICSI, launch workers through SLURM.
        # Otherwise, launch in background.
        print("Starting {} workers...".format(num_workers))
        cmd = "rqworker --burst {}".format(name)
        if util.running_on_icsi():
            host, port = vislab.config['servers']['redis']
            job_log_dirname = util.makedirs(
                vislab.config['paths']['shared_data'] + '/rqworkers')
            cmd = "srun -p vision --cpus-per-task={} --mem={}".format(
                cpus_per_task, mem)
            cmd += " --time={} --output={}/{}_%j-out.txt".format(
                max_time, job_log_dirname, name)
            if len(vislab.config['servers']['redis_exclude']) > 0:
                cmd += " --exclude={}".format(
                    vislab.config['servers']['redis_exclude'])
            cmd += " rqworker --host {} --port {} --burst {}".format(
                host, port, name)
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
            msg = "\r{:.1f} s passed, {} succeeded / {} failed".format(
                time.time() - t, num_succeeded, num_failed)
            msg += " out of {} total".format(len(jobs))
            sys.stdout.write(msg)
            sys.stdout.flush()
            if num_succeeded + num_failed == len(jobs):
                break
            time.sleep(1)
        sys.stdout.write('\n')
        sys.stdout.flush()
    print('Done with all jobs.')

    # Print some statistics about the run.
    failed_jobs = [j for j in fq.get_jobs() if j.origin == name]
    print("{} jobs failed and went into the failed queue.".format(
        len(failed_jobs)))

    # If requested, aggregate and return the results.
    if aggregate:
        results = [job.result for job in jobs]
        if chunk_size > 1:
            # Watch out for some jobs returning None results.
            all_results = []
            for result in results:
                if result is not None:
                    all_results += result
            results = all_results
        return results

    return None
