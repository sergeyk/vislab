"""
Often, we need to populate a database with documents, one per id.
Constructing each document takes some amount of work, and is best done
in parallel.
Due to setup costs, each job should consist of several documents.
This module provides generic implementation of this use case.

Tasks
-----
- make job_info a class?
"""
import time
import importlib
import numpy as np
import rq
import vislab

required_job_info_keys = ['worker_fn', 'db_name', 'collection_name']


def get_collection(job_info):
    client = vislab.util.get_mongodb_client()
    db = client[job_info['db_name']]
    collection = db[job_info['collection_name']]
    return collection


def process_and_insert(query_list, kwargs_list, job_info):
    assert all(key in job_info for key in required_job_info_keys)

    if job_info['module'] is not None:
        module = importlib.import_module(job_info['module'])

    # Get collection
    collection = get_collection(job_info)

    # Aggregate documents for queries that are not already in collection.
    results = []
    for query, kwargs in zip(query_list, kwargs_list):
        if collection.find_one(query) is not None:
            continue
        try:
            worker_fn = getattr(module, job_info['worker_fn'])
            results.append(worker_fn(**kwargs))
        except Exception as e:
            print(e)
    results = filter(lambda x: x is not None, results)

    # Bulk insert documents into collection.
    if len(results) > 0:
        collection.insert(results)


def submit_to_rq(
        query_list, kwargs_list, job_info,
        queue_name='default', job_size=10, timeout=60):
    # Establish connection to the redis queue and clear it.
    redis_conn = vislab.util.get_redis_client()
    q = rq.Queue(queue_name, connection=redis_conn)
    q.empty()

    # Chunk up the jobs and submit them.
    assert len(query_list) == len(kwargs_list)
    num_docs = len(query_list)
    query_list_chunks = np.array_split(query_list, num_docs / job_size)
    kwargs_list_chunks = np.array_split(kwargs_list, num_docs / job_size)
    for query_list, kwargs_list in zip(query_list_chunks, kwargs_list_chunks):
        q.enqueue_call(
            func=process_and_insert,
            args=(query_list, kwargs_list, job_info),
            timeout=timeout
        )

    print("All jobs have been submitted. Please start workers:")
    print("rqworker {}".format(queue_name))

    collection = get_collection(job_info)

    t = time.time()
    while True:
        if collection.count() == num_docs:
            break
        time.sleep(5)
        print('Done with {}/{} images in {:.2f} s'.format(
            collection.count(), num_docs, time.time() - t))

    print("All done")
