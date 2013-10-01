import os
import glob
import xml.dom.minidom as minidom
import pandas as pd
import time
import joblib
import vislab.backend


def load_pascal_annotation(filename):
    """
    Load image and bounding boxes info from the PASCAL VOC XML format.
    """
    def get_data_from_tag(node, tag):
        if tag is "bndbox":
            bbox = node.getElementsByTagName(tag)[0]
            x1 = int(bbox.childNodes[1].childNodes[0].data)
            y1 = int(bbox.childNodes[3].childNodes[0].data)
            x2 = int(bbox.childNodes[5].childNodes[0].data)
            y2 = int(bbox.childNodes[7].childNodes[0].data)
            return (x1, y1, x2, y2)
        else:
            return node.getElementsByTagName(tag)[0].childNodes[0].data

    with open(filename) as f:
        data = minidom.parseString(f.read())
    name = str(get_data_from_tag(data, "filename"))
    name = name[:-4]  # get rid of file extension

    # Load object bounding boxes into a data frame.
    objects_df = pd.DataFrame([
        {
            'class': str(get_data_from_tag(obj, "name")).lower().strip(),
            'pose': str(get_data_from_tag(obj, "pose")).lower().strip(),
            'difficult': int(get_data_from_tag(obj, "difficult")) == 1,
            'truncated': int(get_data_from_tag(obj, "truncated")) == 1,
            'xmin': get_data_from_tag(obj, "bndbox")[0],
            'ymin': get_data_from_tag(obj, "bndbox")[1],
            'xmax': get_data_from_tag(obj, "bndbox")[2],
            'ymax': get_data_from_tag(obj, "bndbox")[3],
        }
        for obj in data.getElementsByTagName("object")
    ])
    objects_df.index = pd.MultiIndex.from_tuples(
        [(name, ind) for ind in objects_df.index],
        names=['image_id', 'object_id']
    )

    # Load misc info about the image into series.
    source = data.getElementsByTagName('source')[0]
    size = data.getElementsByTagName("size")[0]
    classes = objects_df['class'].unique()
    image_info = {
        '_width': int(get_data_from_tag(size, "width")),
        '_height': int(get_data_from_tag(size, "height")),
        '_source': str(get_data_from_tag(source, 'annotation'))
    }
    for class_ in classes:
        image_info[class_] = True
    image_series = pd.Series(image_info, name=name)

    return image_series, objects_df


def load_pascal(force=False):
    """
    Load all the annotations, including object bounding boxes.
    Warning: this takes a good few minutes to load from scratch!
    """
    filename = vislab.backend.config['data_dir'] + '/pascal_dfs.h5'
    if not force and os.path.exists(filename):
        images_df = pd.read_hdf(filename, 'images_df')
        objects_df = pd.read_hdf(filename, 'objects_df')
        return images_df, objects_df

    annotations = glob.glob(vislab.backend.config['VOC_DIR'] +
                            '/Annotations/*.xml')
    t = time.time()
    results = joblib.Parallel(n_jobs=8)(
        joblib.delayed(load_pascal_annotation)(annotation)
        for annotation in annotations
    )
    images, objects_dfs = zip(*results)
    images_df = pd.DataFrame(list(images))

    # Get the canonical split information.
    splits_dir = vislab.backend.config['VOC_DIR'] + '/ImageSets/Main'
    images_df['_split'] = None
    for split in ['train', 'val', 'test']:
        with open(splits_dir + '/{}.txt'.format(split)) as f:
            inds = [x.strip() for x in f.readlines()]
        images_df['_split'].ix[inds] = split

    objects_df = objects_dfs[0]
    for df in objects_dfs[1:]:
        objects_df = objects_df.append(df)
    print('load_pascal: finished in {:.3f} s'.format(time.time() - t))

    images_df.to_hdf(filename, 'images_df', mode='w')
    objects_df.to_hdf(filename, 'objects_df', mode='w')
    return images_df, objects_df
