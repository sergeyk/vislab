# -*- coding: utf-8 -*-
"""
The good old INRIA pedestrian dataset.

I could not download the archive from [1],
and so downloaded the data from our group cluster.

~/work/vision_data/INRIAPerson
├── Test
│   ├── annotations
│   ├── annotations.lst
│   ├── neg
│   ├── neg.lst
│   ├── pos
│   └── pos.lst
└── Train
    ├── annotations
    ├── annotations.lst
    ├── neg
    ├── neg.lst
    ├── pos
    └── pos.lst

[1]: http://pascal.inrialpes.fr/data/human/
"""
import os
import pandas as pd
import vislab

dirname = vislab.config['paths']['INRIAPerson']


def parse_annotation(anno_path):
    with open(dirname + '/' + anno_path) as f:
        lines = f.readlines()

    objects = []
    for line in lines:
        if line.startswith('Image filename'):
            filename = line.split(':')[-1].strip()[1:-1]
            name = filename.split('/')[-1][:-4]
        if line.startswith('Image size'):
            width = int(line.split(':')[-1].split('x')[0].strip())
            height = int(line.split(':')[-1].split('x')[1].strip())
        if line.startswith('# Details for object'):
            objects.append({'class': line.split('(')[-1][1:-3]})
        if line.startswith('Original label for object'):
            objects[-1]['label'] = line.split(':')[-1][2:-2]
        if line.startswith('Center point on object'):
            objects[-1]['center_x'] = int(
                line.split('(')[-1].split(',')[0])
            objects[-1]['center_y'] = int(
                line.split(')')[-2].split(',')[-1].strip())
        if line.startswith('Bounding box for object'):
            _ = line.split(' : ')[-1]
            objects[-1]['xmin'] = int(_.split(' - ')[0].split(',')[0][1:])
            objects[-1]['ymin'] = int(_.split(' - ')[0].split(',')[1][1:-1])
            objects[-1]['xmax'] = int(_.split(' - ')[1].split(',')[0][1:])
            objects[-1]['ymax'] = int(_.split(' - ')[1].split(',')[1][1:-2])

    objects_df = pd.DataFrame(objects)
    objects_df['width'] = width
    objects_df['height'] = height
    objects_df['filename'] = filename
    objects_df.index = pd.MultiIndex.from_tuples(
        [(name, ind) for ind in objects_df.index],
        names=['image_id', 'object_id']
    )
    return objects_df


def load_dataset(force=False):
    cache_filename = vislab.config['paths']['shared_data'] + '/inria_dfs.h5'
    if not force and os.path.exists(cache_filename):
        images_df = pd.read_hdf(cache_filename, 'images_df')
        objects_df = pd.read_hdf(cache_filename, 'objects_df')
        return images_df, objects_df

    objects_dfs = []
    images_dfs = []
    for split in ['Train', 'Test']:
        # Load object data.
        anno_filenames = [
            _.strip() for _
            in open('{}/{}/annotations.lst'.format(dirname, split)).readlines()
        ]
        objects_df = pd.concat((
            parse_annotation(anno_filename)
            for anno_filename in anno_filenames
        ))

        # Construct images_df from the objects data.
        grouped = objects_df.groupby(level=0)
        images_df = pd.DataFrame()
        images_df['filename'] = objects_df.groupby(level=0).first()['filename']
        images_df[['filename', 'width', 'height']] = grouped.first()[
            ['filename', 'width', 'height']]

        # We know that all objects are PASperson, but let's count them.
        images_df['PASperson'] = True
        images_df['num_objects'] = grouped.count()['class']

        # Load negative examples and append to the images_df.
        neg_filenames, neg_image_ids = map(list, zip(*[
            (_.strip(), _.strip().split('/')[-1][:-4]) for _
            in open('{}/{}/neg.lst'.format(dirname, split)).readlines()
        ]))
        neg_images_df = pd.DataFrame(index=neg_image_ids)
        neg_images_df['filename'] = neg_filenames
        neg_images_df['PASperson'] = False
        neg_images_df['num_objects'] = 0
        images_df = images_df.append(neg_images_df)

        objects_df['split'] = split
        images_df['split'] = split

        objects_dfs.append(objects_df)
        images_dfs.append(images_df)

    objects_df = pd.concat(objects_dfs)
    images_df = pd.concat(images_dfs)

    images_df.to_hdf(cache_filename, 'images_df', mode='w')
    objects_df.to_hdf(cache_filename, 'objects_df', mode='a')

    return images_df, objects_df
