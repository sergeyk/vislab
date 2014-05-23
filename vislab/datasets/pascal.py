"""
PASCAL VOC: http://pascallin.ecs.soton.ac.uk/challenges/VOC/

Everything loaded from files, and images distributed with dataset.
Nothing special to do in parallel.
"""
import os
import numpy as np
import glob
import xml.dom.minidom as minidom
import pandas as pd
import time
import joblib
import operator
import multiprocessing
import vislab

pascal_classes = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]


def get_image_filename_for_id(image_id):
    return '{}/JPEGImages/{}.jpg'.format(
        vislab.config['paths']['VOC'], image_id)


def get_metaclass_df(VOCyear='VOC2012', force=False, args=None):
    """
    Load the image classification data, computing 'metaclass' labels.
    """
    label_df, objects_df = load_pascal(VOCyear, force, args)
    label_df = label_df.fillna(False)

    # Group classes into metaclasses.
    metaclasses = {
        'metaclass_animal': [
            'bird', 'cat', 'cow', 'dog', 'horse', 'sheep'
        ],
        'metaclass_vehicle': [
            'aeroplane', 'bicycle', 'boat',
            'bus', 'car', 'motorbike', 'train'
        ],
        'metaclass_indoor': [
            'bottle', 'chair', 'diningtable',
            'pottedplant', 'sofa', 'tvmonitor'
        ],
        'metaclass_person': [
            'person'
        ]
    }
    for metaclass, classes in metaclasses.iteritems():
        label_df[metaclass] = reduce(
            operator.or_, [label_df[c] for c in classes])

    for class_ in pascal_classes:
        del label_df[class_]

    label_df['image_filename'] = [
        '{}/JPEGImages/{}.jpg'.format(vislab.config['paths'][VOCyear], _)
        for _ in label_df.index
    ]
    del label_df['_split']
    return label_df


def get_class_df(VOCyear='VOC2012', force=False, args=None):
    """
    Load the image classification data, with metaclasses.
    """
    label_df, objects_df = load_pascal(VOCyear, force, args)
    label_df = label_df.fillna(False)

    # Append class_ before every class.
    label_df.columns = [
        'class_' + col if col in pascal_classes
        else col
        for col in label_df.columns
    ]
    label_df['image_filename'] = [
        '{}/JPEGImages/{}.jpg'.format(vislab.config['paths'][VOCyear], _)
        for _ in label_df.index
    ]
    del label_df['_split']
    return label_df


def get_det_df(VOCyear='VOC2012', force=False, args=None):
    """
    Load the image classification data, with metaclasses.
    """
    _, objects_df = load_pascal(VOCyear, force, args)
    return objects_df


def load_annotation_files(filenames, num_workers=1):
    t = time.time()
    if num_workers > 1:
        results = joblib.Parallel(n_jobs=num_workers)(
            joblib.delayed(_load_pascal_annotation)(fname)
            for fname in filenames
        )
    else:
        results = [_load_pascal_annotation(fname) for fname in filenames]
    images, objects_dfs = zip(*results)
    images_df = pd.DataFrame(list(images))
    objects_df = pd.concat(objects_dfs)
    print('load_annotation_files: finished in {:.3f} s'.format(
        time.time() - t))
    return images_df, objects_df


def load_pascal(VOCyear='VOC2012', force=False, args=None):
    """
    Load all the annotations, including object bounding boxes.
    Loads XML data in args['num_workers'] threads using joblib.Parallel.

    Warning: this takes a few minutes to load from scratch!
    """
    if args is None:
        args = {'num_workers': multiprocessing.cpu_count()}

    cache_filename = \
        vislab.config['paths']['shared_data'] + \
        '/pascal_{}_dfs.h5'.format(VOCyear)
    if not force and os.path.exists(cache_filename):
        images_df = pd.read_hdf(cache_filename, 'images_df')
        objects_df = pd.read_hdf(cache_filename, 'objects_df')
        return images_df, objects_df

    # Load all annotation file data (should take < 30 s).
    annotation_filenames = glob.glob(
        vislab.config['paths'][VOCyear] + '/Annotations/*.xml')
    images_df, objects_df = load_annotation_files(
        annotation_filenames, args['num_workers'])

    # Get the split information.
    splits_dir = vislab.config['paths'][VOCyear] + '/ImageSets/Main'
    images_df['_split'] = None
    for split in ['train', 'val', 'test']:
        split_filename = splits_dir + '/{}.txt'.format(split)
        if not os.path.exists(split_filename):
            print("{} split does not exist".format(split))
            continue
        with open(split_filename) as f:
            inds = [x.strip() for x in f.readlines()]
        safe_inds = set(inds).intersection(images_df.index)
        images_df['_split'].ix[safe_inds] = split

    # Drop images without a split (VOC2007 images in the VOC2012 set).
    images_df = images_df.dropna(subset=['_split'])

    # Drop corresponding images in the objects_df.
    objects_df = objects_df.ix[images_df.index]

    # Propagate split info to objects_df
    objects_df['split'] = np.repeat(
        images_df['_split'].values, images_df['_num_objects'].values)

    # Make sure that all labels are either True or False.
    images_df = images_df.fillna(False)

    images_df.to_hdf(cache_filename, 'images_df', mode='w')
    objects_df.to_hdf(cache_filename, 'objects_df', mode='a')
    return images_df, objects_df


def _load_pascal_annotation(filename):
    """
    Load image and bounding boxes info from XML file in the PASCAL VOC
    format.
    """
    def get_data_from_tag(node, tag):
        try:
            return node.getElementsByTagName(tag)[0].childNodes[0].data
        except:
            return 0

    with open(filename) as f:
        data = minidom.parseString(f.read())
    name = str(get_data_from_tag(data, "filename"))
    if name[-4:] == '.jpg':
        name = name[:-4]

    # Load object bounding boxes into a data frame.
    all_obj_data = []
    for obj in data.getElementsByTagName("object"):
        obj_data = {
            # It's unclear whether these coordinates are 1-based.
            # For PASCAL, I think they are, but for Imagenet, they're
            # not. It shouldn't matter.
            'xmin': float(get_data_from_tag(obj, "xmin")),
            'ymin': float(get_data_from_tag(obj, "ymin")),
            'xmax': float(get_data_from_tag(obj, "xmax")),
            'ymax': float(get_data_from_tag(obj, "ymax")),
            'class': str(get_data_from_tag(obj, "name")).lower().strip()
        }
        if obj.getElementsByTagName('pose'):
            obj_data['pose'] = str(
                get_data_from_tag(obj, "pose")).lower().strip()
        if obj.getElementsByTagName('difficult'):
            obj_data['difficult'] = int(
                get_data_from_tag(obj, "difficult")) == 1
        if obj.getElementsByTagName('truncated'):
            obj_data['truncated'] = int(
                get_data_from_tag(obj, "truncated")) == 1

        all_obj_data.append(obj_data)
    objects_df = pd.DataFrame(all_obj_data)

    # Load size info
    size = data.getElementsByTagName("size")[0]
    image_info = {
        '_width': int(get_data_from_tag(size, "width")),
        '_height': int(get_data_from_tag(size, "height")),
        '_num_objects': objects_df.shape[0]
    }

    # Set index and size info of objects_df (if there are objects).
    if objects_df.shape[0] > 0:
        objects_df.index = pd.MultiIndex.from_tuples(
            [(name, ind) for ind in objects_df.index],
            names=['image_id', 'object_id']
        )
        objects_df['width'] = image_info['_width']
        objects_df['height'] = image_info['_height']

    # Load misc info about the image into series.
    source = data.getElementsByTagName('source')[0]
    if source.getElementsByTagName('annotation'):
        image_info['_source'] = str(get_data_from_tag(source, 'annotation'))
    if 'class' in objects_df.columns:
        classes = objects_df['class'].unique()
        for class_ in classes:
            image_info[class_] = True
    image_series = pd.Series(image_info, name=name)

    return image_series, objects_df


if __name__ == '__main__':
    get_class_df(VOCyear='VOC2012', force=True)
