"""
An implementation of a simple work/results queue using Redis,
based on http://flask.pocoo.org/snippets/73/
"""
import time
import uuid
import cPickle
from vislab import util


class DelayedResult(object):
    """
    Simple class to get and save the return value of a job.
    """
    def __init__(self, key, redis_conn):
        self.key = key
        self.redis_conn = redis_conn
        self._rv = None

    @property
    def return_value(self):
        if self._rv is None:
            rv = self.redis_conn.get(self.key)
            if rv is not None:
                self._rv = cPickle.loads(rv)
                del self.redis_conn
        return self._rv


def submit_job(function_name, kwargs, queue_name='default'):
    """
    Parameters
    ----------
    function_name: string
        Name of the function that will be expected
        by the waiting process.
    kwargs: dict
        Keyword arguments to pass to the function.
    queue_name: string ['default']
        Name of the queue to wait on.
    """
    redis_conn = util.get_redis_client()
    key = '%s:result:%s' % (queue_name, str(uuid.uuid4()))
    redis_conn.rpush(
        queue_name, cPickle.dumps((function_name, kwargs, key)))
    return DelayedResult(key, redis_conn)


def get_return_value(job, poll_interval=0.05, timeout=30):
    """
    Poll the job until it returns a non-None value or an Exception, or
    the polling times out.

    Parameters
    ----------
    job: DelayedJob
    poll_interval: float [.05]
        In seconds.
    timeout: float [30]
        In seconds.
    """
    t = time.time()
    while job.return_value is None and (time.time() - t) < timeout:
        time.sleep(poll_interval)
    if isinstance(job.return_value, Exception):
        raise job.return_value
    print("get_return_value: returning after {:.3f} s".format(
        time.time() - t))
    return job.return_value


def poll_for_jobs(registered_functions, queue_name='default', rv_ttl=60):
    """
    Poll the given queue for jobs, complete it if it matches one of the
    registered functions, and place return value of the function call
    on the results queue (given as part of the job).

    Parameters
    ----------
    registered_functions: dict
        Mapping from a name that is received as part of the job to
        actual bound function to call.
    queue_name: string ['default']
        Name of the queue to wait on.
    rv_ttl: float [60]
        Result's time to live, in seconds.
    """
    redis_conn = util.get_redis_client()
    print("poll_for_jobs: now listening on {}".format(queue_name))
    while True:
        msg = redis_conn.blpop(queue_name)
        function_name, kwargs, key = cPickle.loads(msg[1])
        try:
            assert(function_name in registered_functions)
            rv = registered_functions[function_name](**kwargs)
        except Exception as e:
            rv = e
        if rv is not None:
            redis_conn.set(key, cPickle.dumps(rv))
            redis_conn.expire(key, rv_ttl)
