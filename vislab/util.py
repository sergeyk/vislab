import os
import pandas as pd
import pymongo
import redis
import socket
import tempfile
import cPickle
import subprocess


def add_cmdline_args(parser):
    """
    Add commonly used command line arguments to given ArgumentParser.
    """
    parser.add_argument(
        '--random_seed', default=42)


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

# Gather data by calling generator_fn into "filename", if it doesn't already exist.
# If it does exist, just reload it from the file.
# Use force=True to force regenerating the data.
def load_or_generate_df(filename, generator_fn, force=False):
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


def print_collection_counts():
    """
    Print all collections and their counts for all databases in MongoDB.
    """
    client = get_mongodb_client()
    for db_name in client.database_names():
        for coll_name in client[db_name].collection_names():
            print db_name, coll_name, client[db_name][coll_name].count()


def get_redis_conn():
    host = 'flapjack' if running_on_icsi() else 'localhost'
    redis_conn = redis.Redis(host)
    return redis_conn


def pickle_function_call(func_name, args):
    f, temp_filename = tempfile.mkstemp()
    with open(temp_filename, 'w') as f:
        cPickle.dump((func_name, args), f)
    c = "import os; import cPickle;"
    c += "f = open('{0}'); func, args = cPickle.load(f); f.close();"
    c += "os.remove('{0}'); func(*args)"
    c = c.format(temp_filename)
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
