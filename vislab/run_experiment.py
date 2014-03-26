"""
Generate list of commands to run prediction experiments on
AVA, Flickr, and Wikipaintings datasets.
"""
import argparse
import itertools


def cmds_for_experiment(experiment_name, args):
    ## AVA
    # AVA Aesthetic experiment replicates the CVPR 12 paper evaluation
    # vs number of images used and 'delta' (threshold of positive clf).
    if experiment_name == 'flickr':
        flags = '--dataset=flickr --collection_name=flickr_mar23'
        prediction_labels = [['style_*']]

    elif experiment_name == 'pinterest_80k':
        flags = '--dataset=pinterest_80k --collection_name=pinterest_80k_mar23'
        prediction_labels = [['style_*']]

    elif experiment_name == 'wikipaintings':
        flags = '--dataset=wikipaintings --collection_name=wikipaintings_mar23'
        prediction_labels = [['style_*']]

    elif experiment_name == 'pascal':
        flags = '--dataset=pascal --collection_name=pascal_mar23'
        prediction_labels = [['class_*']]

    elif experiment_name == 'pascal_mc':
        flags = '--dataset=pascal_mc --collection_name=pascal_mc_mar23'
        prediction_labels = [['metaclass_*']]

    elif experiment_name == 'pascal_mc_on_wikipaintings':
        flags = '--dataset=wikipaintings --source_dataset=pascal_mc'
        flags += ' --collection_name=pascal_mc_on_wikipaintings_mar23'
        prediction_labels = [['metaclass_*']]

    elif experiment_name == 'pascal_mc_on_flickr':
        flags = '--dataset=flickr --source_dataset=pascal_mc'
        flags += ' --collection_name=pascal_mc_on_flickr_mar23'
        prediction_labels = [['metaclass_*']]

    elif experiment_name == 'pascal_mc_on_pinterest_80k':
        flags = '--dataset=pinterest_80k --source_dataset=pascal_mc'
        flags += ' --collection_name=pascal_mc_on_pinterest_80k_mar23'
        prediction_labels = [['metaclass_*']]

    elif experiment_name == 'pinterest_80k_on_flickr':
        flags = '--dataset=flickr --source_dataset=pinterest_80k'
        flags += ' --collection_name=pinterest_80k_on_flickr_mar23'
        prediction_labels = [['"style_*"']]

    elif experiment_name == 'flickr_on_pinterest_80k':
        flags = '--dataset=pinterest_80k --source_dataset=flickr'
        flags += ' --collection_name=flickr_on_pinterest_80k_mar23'
        prediction_labels = [['"style_*"']]

    elif experiment_name == 'flickr_on_pascal_mc':
        flags = '--dataset=pascal --source_dataset=flickr'
        flags += ' --collection_name=flickr_on_pascal_mar23'
        prediction_labels = [['"style_*"']]

    elif experiment_name == 'flickr_on_pascal':
        flags = '--dataset=pascal_mc --source_dataset=flickr'
        flags += ' --collection_name=flickr_on_pascal_mc_mar23'
        prediction_labels = [['"style_*"']]

    else:
        raise ValueError("Unknown experiment_name")

    # Set the features to use in experiments.
    #
    # The first element of the tuple can be a comma-separated list of
    # features.'
    # The second element is either an empty string or 'all', which
    # activates quadratic expansion of the features in the list.
    feature_quadratic = [
        # ('noise', ''),
        #('gist_256', ''),
        #('lab_hist', ''),
        #('gbvs_saliency', ''),
        # ('mc_bit', ''),
        ('caffe_fc6', ''),
        # ('caffe_fc7', ''),
        #('fusion_ava_style_oct22', '')
    ]

    # Do some list munging to flatten the settings correctly.
    settings = list(itertools.product(feature_quadratic, prediction_labels))
    settings = [
        [val for subl in setting for val in subl]
        for setting in settings
    ]

    # For each setting, form the command.
    cmds = []
    for setting in settings:
        features, quadratic, prediction_label = setting

        # Thanks to VW tmemory handling and unix pipes everywhere, memory
        # requirements are crazy low. But IO requirements are high, so
        # still have to be careful when submitting concurrent jobs.
        #
        # With 24 bits, at most 360m VM is consumed per VW instance.
        # We're using 18 bits, so it's like 8m.
        # But the most memory is used when caching, and that's done by 3
        # workers, so let's go with at least 9 GB.
        mem = max(720 * args.num_workers, 9000)

        python_cmd = 'python vislab/predict.py predict {}'.format(
            flags)
        python_cmd += ' --prediction_label="{}" --features="{}"'.format(
            prediction_label, features)
        python_cmd += ' --mem={} --num_workers={}'.format(
            mem, args.num_workers)

        if len(quadratic) > 0:
            python_cmd += ' --quadratic'

        if args.slurm:
            max_time = '9:0:0'
            job_output_dirname = 'data/shared/predict/workers'
            cmd = "srun -p vision"
            if len(args.exclude) > 0:
                cmd += " --exclude={}".format(args.exclude)
            cmd += " --mem={} --cpus-per-task={} --time={}".format(
                mem, int(round(1.2 * args.num_workers)), max_time)

            cmd += " -o {}/%j.txt {} &".format(job_output_dirname, python_cmd)
        else:
            cmd = python_cmd

        cmds.append(cmd)

    return cmds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_workers',
        help="number of workers to use in processing jobs",
        type=int, default=8)
    parser.add_argument(
        '--slurm', action='store_true')
    parser.add_argument(
        '--exclude', type=str, default='',
        help="Comma-separated list of nodes to exclude from slurm queueing.")
    parser.add_argument(
        '--experiments', type=str, default='', required=True,
        help="Comma-separated list of experiments to generate commands for.")
    args = parser.parse_args()
    args.experiments = args.experiments.split(',')

    for experiment_name in args.experiments:
        for cmd in cmds_for_experiment(experiment_name, args):
            print cmd
