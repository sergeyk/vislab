"""
Utilities and glue code for providing a command-line interface to
module functions.
"""
import sys
import argparse


def add_cmdline_args(group_name, parser):
    """
    All module-specific command-line arguments are specified here, split
    into groups by functionality.
    Any command-line interfacing function can include any subset of
    these groups.
    """
    if group_name == 'common':
        parser.add_argument(
            '--force',
            action="store_true",
            default=False)
        parser.add_argument(
            '--random_seed',
            type=int,
            default=42)

    # Stuff for selecting the dataset and limiting the number of images:
    elif group_name == 'dataset':
        parser.add_argument(
            '--dataset',
            help="select which dataset to use",
            required=True)
        parser.add_argument(
            '--source_dataset',
            help="select which dataset to use as trained clf source",
            type=str,
            default=None)
        parser.add_argument(
            '--num_images',
            help="number of images to use from the dataset (-1 for all)",
            type=int,
            default=-1)
        parser.add_argument(
            '--force_dataset',
            help="Force reloading of dataset from scratch",
            action="store_true",
            default=False)

    # Stuff for selecting list of features and forcing their overwrite.
    elif group_name == 'feature':
        parser.add_argument(
            '--features',
            help="Comma-separated list of FEATURES from vislab/feature.py",
            default='size')
        parser.add_argument(
            '--standardize',
            help="Standardize features during caching?",
            action="store_true",
            default=False)
        parser.add_argument(
            '--force_features',
            help="force overwrite of existing features",
            action="store_true",
            default=False)

    # Stuff for distributed computation:
    elif group_name == 'processing':
        parser.add_argument(
            '--num_workers',
            help="number of workers to use in processing jobs",
            type=int,
            default=1)
        parser.add_argument(
            '--chunk_size',
            help="number of jobs to assign to a worker at once",
            type=int,
            default=20)
        parser.add_argument(
            '--mem',
            help="amount of memory that a single worker will use",
            type=int,
            default=3000)
        parser.add_argument(
            '--cpus_per_task',
            help="number of cpus that a single worker will use",
            type=int,
            default=4)

    # Forming a prediction dataset and setting properties of predictor.
    elif group_name == 'prediction':
        parser.add_argument(
            '--prediction_label',
            required=True,
            help="""
column of the dataframe to use for label.
can contain wildcard characters,
i.e. 'style_*' will match multiple columns""")
        parser.add_argument(
            '--collection_name',
            help="name of the collection to write prediction results to",
            default="default")
        parser.add_argument(
            '--test_frac',
            help="fraction of dataset to use for testing",
            type=float,
            default=0.2)
        parser.add_argument(
            '--balanced',
            help="should the validation set be balanced for multiclass",
            action="store_true",
            default=False)
        parser.add_argument(
            '--min_pos_frac',
            help="minimum fraction of positive examples in training",
            type=float,
            default=0.1)
        parser.add_argument(
            '--quadratic',
            help="perform quadratic expansion of the features",
            type=str,
            default=None)
        parser.add_argument(
            '--bit_precision',
            help="bit precision of the VW classifier",
            type=int,
            default=18)
        parser.add_argument(
            '--force_predict',
            help="force overwrite of existing results",
            action="store_true",
            default=False)
        parser.add_argument(
            '--ava_num_train',
            help="number of training images to use",
            type=int,
            default=-1)
        parser.add_argument(
            '--ava_delta',
            help="AVA: only use images >= delta from mean rating",
            type=float,
            default=0.0)

    else:
        raise Exception("Unknown group!")


def get_args(script_name, calling_function_name, groups=None):
    """
    Parse and return all options and arguments.

    Parameters
    ----------
    calling_function_name: string
    groups: list
        List of groups of options to include.

    Returns
    -------
    args: argparse.Namespace
    """
    usage = "python {} {} [options]".format(
        script_name, calling_function_name)
    parser = argparse.ArgumentParser(usage)

    all_groups = [
        'common', 'processing', 'dataset', 'feature', 'prediction'
    ]
    if groups is None:
        groups = all_groups

    # Common arguments are always part of the argument parser.
    if 'common' not in groups:
        groups.append('common')

    for group in groups:
        add_cmdline_args(group, parser)

    # Get the parsed options and arguments, keeping in mind that the
    # first argument is the name of the calling function.
    parser.add_argument('function', nargs=1)
    args = parser.parse_args()

    # Split features into a list.
    if 'features' in args:
        args.features = args.features.split(',')
    return args


def run_function_in_file(name, possible_functions):
    """
    Provide a command line interface to call a function in a file.
    Simply call the function named as the first commandline argument.

    Parameters
    ----------
    name: string
        Name of the file that is calling this method.
        What the user executed.
    possible_functions: dict
        name: function
    """
    def print_usage():
        print("usage:\tpython {} <function> [args] <args>".format(name))
        print("possible functions:")
        for func in possible_functions.keys():
            print("\t" + func)
        sys.exit(1)

    if len(sys.argv) < 2:
        print_usage()

    selected_function = sys.argv[1]
    if selected_function in possible_functions:
        possible_functions[selected_function]()
    else:
        print("Unknown function: {}".format(selected_function))
        print_usage()
