import time
import uuid
import cPickle
from vislab.backend import util


class DelayedResult(object):
    def __init__(self, key):
        # TODO: maybe pass in redis_conn to here?
        self.key = key
        self._rv = None

    @property
    def return_value(self):
        # TODO: init in __init__
        self.redis_conn = util.get_redis_conn()
        if self._rv is None:
            rv = self.redis_conn.get(self.key)
            if rv is not None:
                self._rv = cPickle.loads(rv)
                del self.redis_conn
        return self._rv


def redis_delay(fn, kwargs, qkey='default'):
    redis_conn = util.get_redis_conn()
    key = '%s:result:%s' % (qkey, str(uuid.uuid4()))
    s = cPickle.dumps((fn, kwargs, key))
    redis_conn.rpush(qkey, s)
    return DelayedResult(key)


def get_return_value(job):
    """
    Keep polling job until it returns a non-None value or an Exception.
    """
    while job.return_value is None:
        time.sleep(.1)
    if isinstance(job.return_value, Exception):
        raise job.return_value
    return job.return_value


def wait_for_job(fn, qkey='default'):
    redis = util.get_redis_conn()
    rv_ttl = 60
    while True:
        msg = redis.blpop(qkey)
        method_name, kwargs, key = cPickle.loads(msg[1])
        try:
            rv = fn(method_name, kwargs)
        except Exception as e:
            rv = e
        if rv is not None:
            redis.set(key, cPickle.dumps(rv))
            redis.expire(key, rv_ttl)
