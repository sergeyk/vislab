import os
import pandas as pd
import pymongo
import redis
import socket
import tempfile
import cPickle
import subprocess
import shutil
import vislab


def zero_results(collection, query):
    return collection.find(query).limit(1).count() == 0


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


def load_or_generate_df(filename, generator_fn, force=False, args=None):
    """
    If filename does not already exist, gather data with generator_fn,
    and write to filename.
    If filename does exist, load from it.
    """
    if not force and os.path.exists(filename):
        df = pd.read_hdf(filename, 'df')
    else:
        df = generator_fn(args)
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
        host, port = vislab.config['servers']['mongo']
        connection = pymongo.MongoClient(host, port)
    except pymongo.errors.ConnectionFailure:
        raise Exception(
            "Need a MongoDB server running on {}, port {}".format(host, port))
    return connection


def print_collection_counts():
    """
    Print all collections and their counts for all databases in MongoDB.
    """
    client = get_mongodb_client()
    for db_name in client.database_names():
        for coll_name in client[db_name].collection_names():
            print('{} |\t\t{}: {}'.format(
                db_name, coll_name, client[db_name][coll_name].count()))


def get_redis_client():
    host, port = vislab.config['servers']['redis']
    try:
        connection = redis.Redis(host, port)
        connection.ping()
    except redis.ConnectionError:
        raise Exception(
            "Need a Redis server running on {}, port {}".format(host, port))
    return connection


def pickle_function_call(func_name, args):
    f, temp_filename = tempfile.mkstemp()
    with open(temp_filename, 'w') as f:
        cPickle.dump((func_name, args), f)
    c = "import os; import cPickle;"
    c += "f = open('{0}'); func, args = cPickle.load(f); f.close();"
    c += "os.remove('{0}'); func(*args)"
    c = c.format(temp_filename)
    return c


def run_through_bash_script(cmds, filename=None, verbose=False, num_workers=1):
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
    num_workers: int [1]
        If > 1, commands are piped through parallel -j num_workers
    """
    assert(num_workers > 0)

    remove_file = False
    if filename is None:
        f, filename = tempfile.mkstemp()
        remove_file = True

    if num_workers > 1:
        contents = "echo \"{}\" | parallel --env PATH -j {}".format(
            '\n'.join(cmds), num_workers)
    else:
        contents = '\n'.join(cmds)

    with open(filename, 'w') as f:
        f.write(contents + '\n')

    if verbose:
        print("Contents of script file about to be run:")
        print(contents)

    p = subprocess.Popen(['bash', filename])
    out, err = p.communicate()

    if remove_file:
        os.remove(filename)
    if not p.returncode == 0:
        print(out)
        print(err)
        raise Exception("Script exited with code {}".format(p.returncode))


def run_shell_cmd(cmd, echo=True):
    """
    Run a command in a sub-shell, capturing stdout and stderr
    to temporary files that are then read.
    """
    _, stdout_f = tempfile.mkstemp()
    _, stderr_f = tempfile.mkstemp()

    print("Running command")
    print(cmd)
    p = subprocess.Popen(
        '{} >{} 2>{}'.format(cmd, stdout_f, stderr_f), shell=True)
    p.wait()

    with open(stdout_f) as f:
        stdout = f.read()
    os.remove(stdout_f)

    with open(stderr_f) as f:
        stderr = f.read()
    os.remove(stderr_f)

    if echo:
        print("stdout:")
        print(stdout)
        print("stderr:")
        print(stderr)

    return stdout, stderr


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


def cleardirs(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    return makedirs(dirname)
