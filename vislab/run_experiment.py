"""
Generate list of commands to run prediction experiments on
AVA, Flickr, and Wikipaintings datasets.
"""
import argparse
import itertools
import vislab.datasets


def cmds_for_experiment(experiment_name, args):
    ## AVA
    # AVA Aesthetic experiment replicates the CVPR 12 paper evaluation
    # vs number of images used and 'delta' (threshold of positive clf).
    if experiment_name == 'ava':
        # # TODO: Currently, not using num_test, num_train, delta.
        # num_test = 50000
        # flags = '--collection_name=ava'
        # num_train_list = [5000, 25000, 50000, 100000,  150000, 200000]
        # deltas = [0, 1]
        # tasks = [
        #     ['clf_rating_mean'],
        #     ['clf_rating_std'],
        #     ['regr_rating_mean'],
        #     ['regr_rating_std']
        # ]
        raise NotImplementedError()

    ## AVA STYLE
    # AVA Style experiment runs on a ~25K subset of AVA that has style
    # labels. The predicted labels are the style labels and the
    # mean and std of the rating distribution. On the latter, we do both
    # classification and regression.
    elif experiment_name == 'ava_aesth':
        ratings_df = vislab.datasets.ava.get_ratings_df()

        flags = '--dataset=ava_style --collection_name=ava_style_aesth_oct29'
        prediction_labels = [
            [x] for x in ratings_df.columns if x.endswith('_bin')
        ]

    elif experiment_name == 'ava_style':
        style_df = vislab.datasets.ava.get_style_df()

        flags = '--dataset=ava_style --collection_name=ava_style_oct17'
        flags = '--dataset=ava_style --collection_name=ava_style_split_oct17'
        flags = '--dataset=ava_style --collection_name=ava_style_oct21'
        prediction_labels = [
            [x] for x in style_df.columns if not x.startswith('_')
        ]

    elif experiment_name == 'pascal_on_ava_style':
        flags = '--dataset=ava_style --source_dataset=pascal --collection_name=pascal_meta_on_ava_oct29'
        prediction_labels = [
            [x] for x in [
                'metaclass_person', 'metaclass_animal',
                'metaclass_indoor', 'metaclass_vehicle',
                'class_dog', 'class_cat', 'class_horse', 'class_car', 'class_bicycle', 'class_bird'
            ]
        ]

    elif experiment_name == 'flickr_on_ava_style':
        flags = '--dataset=ava_style --source_dataset=flickr --collection_name=flickr_on_ava_style_oct30'
        prediction_labels = [['"style_*"']]

    elif experiment_name == 'wikipaintings_on_ava_style':
        flags = '--dataset=ava_style --source_dataset=wikipaintings --collection_name=wp_on_ava_style_oct30'
        prediction_labels = [['"style_*"']]

    ## FLICKR STYLE
    # Flickr experiment runs on a ~50K Flickr style groups dataset.
    elif experiment_name == 'flickr':
        df = vislab.datasets.flickr.load_flickr_df()
        flags = '--dataset=flickr --collection_name=aug29'
        prediction_labels = [
            [x] for x in df.columns if not x.startswith('_')
        ]

    elif experiment_name == 'pascal_on_flickr':
        flags = '--dataset=flickr --source_dataset=pascal --collection_name=pascal_on_flickr_oct29'
        prediction_labels = [
            [x] for x in [
                'metaclass_person', 'metaclass_animal',
                'metaclass_indoor', 'metaclass_vehicle',
                'class_dog', 'class_cat', 'class_horse', 'class_car', 'class_bicycle', 'class_bird'
            ]
        ]

    elif experiment_name == 'wikipaintings_on_flickr':
        flags = '--dataset=flickr --source_dataset=wikipaintings --collection_name=wp_on_flickr_oct30'
        prediction_labels = [['"style_*"']]

    ## WIKIPAINTINGS
    # Wikipaintings experiment runs on a ~100K set of paintings crawled
    # from wikipaintings.org, classifying style labels.
    elif experiment_name == 'wikipaintings_style':
        style_df = vislab.datasets.wikipaintings.get_style_df()
        flags = '--dataset=wikipaintings --collection_name=wikipaintings_sep26'
        prediction_labels = [
            [x] for x in style_df.columns if not x.startswith('_')
        ]

    elif experiment_name == 'pascal_on_wikipaintings':
        flags = '--dataset=wikipaintings --source_dataset=pascal --collection_name=pascal_on_wp_oct29'
        prediction_labels = [
            [x] for x in [
                'metaclass_person', 'metaclass_animal',
                'metaclass_indoor', 'metaclass_vehicle',
                'class_dog', 'class_cat', 'class_horse', 'class_car', 'class_bicycle', 'class_bird'
            ]
        ]

    elif experiment_name == 'flickr_on_wikipaintings':
        flags = '--dataset=wikipaintings --source_dataset=flickr --collection_name=flickr_on_wp_oct30'
        prediction_labels = [['"style_*"']]

    ## PASCAL
    elif experiment_name == 'pascal_metaclass':
        df = vislab.datasets.pascal.get_clf_df()
        flags = '--dataset=pascal --collection_name=pascal_mc_oct16'
        prediction_labels = [
            [x] for x in df.columns if x.startswith('metaclass_')
        ]

    elif experiment_name == 'pascal':
        df = vislab.datasets.pascal.get_clf_df()
        flags = '--dataset=pascal --collection_name=pascal_oct16'
        flags = '--dataset=pascal --collection_name=pascal_oct22'
        flags = '--dataset=pascal --collection_name=pascal_oct29'
        prediction_labels = [
            [x] for x in df.columns if x.startswith('class_')
        ]

    elif experiment_name == 'flickr_on_pascal':
        flags = '--dataset=pascal --source_dataset=flickr --collection_name=flickr_on_pascal_oct30'
        prediction_labels = [['"style_*"']]

    elif experiment_name == 'wikipaintings_on_pascal':
        flags = '--dataset=pascal --source_dataset=wikipaintings --collection_name=wp_on_pascal_oct30'
        prediction_labels = [['"style_*"']]

    elif experiment_name == 'behance_illustration':
        flags = '--dataset=behance_illustration --collection_name=behance_dec28'
        prediction_labels = [[x] for x in [
            'tag_3d', 'tag_animals', 'tag_city', 'tag_fantasy', 'tag_food',
            'tag_girl', 'tag_ink', 'tag_lettering', 'tag_logo', 'tag_minimal',
            'tag_nature', 'tag_pencil', 'tag_portrait', 'tag_retro',
            'tag_skull', 'tag_surreal', 'tag_vector', 'tag_vintage',
            'tag_watercolor', 'tag_wood'
        ]]

    else:
        raise ValueError("Unknown experiment_name")

    # Set the features to use in experiments.
    #
    # The first element of the tuple can be a comma-separated list of
    # features.'
    # The second element is either an empty string or 'all', which
    # activates quadratic expansion of the features in the list.
    feature_quadratic = [
        #('noise', ''),
        #('mc_bit', ''),
        #('gist_256', ''),
        #('lab_hist', ''),
        #('gbvs_saliency', ''),
        ('decaf_fc6', ''),
        #('decaf_fc6_flatten', ''),
        #('decaf_imagenet', ''),
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

        # Thanks to VW memory handling and unix pipes everywhere, memory
        # requirements are crazy low. But IO requirements are high, so
        # still have to be careful when submitting concurrent jobs.
        #
        # With 24 bits, at most 360m VM is consumed per VW instance.
        # We're using 18 bits, so it's like 8m.
        # But the most memory is used when caching, and that's done by 3
        # workers, so let's go with at least 12 GB.
        # We'll go with 2x that to be safe.
        mem = max(720 * args.num_workers, 12000)

        python_cmd = "python vislab/predict.py predict {}".format(
            flags)
        python_cmd += " --prediction_label={} --features={}".format(
            prediction_label, features)
        python_cmd += " --mem={} --num_workers={}".format(
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
        type=int, default=9)
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
