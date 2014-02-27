import vislab.utils.cmdline
import vislab.dataset
import vislab._feature


## Command-line stuff
def compute(args=None):
    """
    Extract features of the requested type for all images in AVA.
    """
    if args is None:
        args = vislab.utils.cmdline.get_args(
            'feature', 'compute', ['dataset', 'processing', 'feature'])
    df = vislab.dataset.get_df_with_args(args)

    for feature in args.features:
        vislab._feature.extract_features(
            df, args.dataset, feature, args.force_features,
            args.mem, args.cpus_per_task, args.num_workers)


def cache_to_h5(args=None):
    """
    Output features in the database for the ids in the loaded dataset to
    HDF5 cache file, one for each type of feature.
    """
    _cache('h5', args)


def cache_to_vw(args=None):
    """
    Output features in the database for the ids in the loaded dataset to
    VW format gzip file, one for each type of feature.
    """
    _cache('vw', args)


def _cache(format, args=None):
    if format == 'h5':
        caching_fn = vislab._feature._cache_to_h5
    elif format == 'vw':
        caching_fn = vislab._feature._cache_to_vw
    else:
        raise Exception("Unknown cache format.")

    if args is None:
        args = vislab.utils.cmdline.get_args(
            'feature', 'cache_to_h5', ['dataset', 'processing', 'feature'])
    df = vislab.dataset.get_df_with_args(args)
    image_ids = df.index.tolist()

    for feature in args.features:
        caching_fn(args.dataset, image_ids, feature, args.force_features)


if __name__ == '__main__':
    possible_functions = {
        'compute': compute,
        'cache_to_h5': cache_to_h5,
        'cache_to_vw': cache_to_vw,
    }
    vislab.utils.cmdline.run_function_in_file(__file__, possible_functions)
