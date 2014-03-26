"""
Copyright Sergey Karayev / Adobe - 2013.
Written during internship at Adobe CTL, San Francisco.

TODO:
- be robust to failures in computing some features in gbvs_saliency
and lab_hist
"""
import os
import numpy as np
from PIL import Image
import leargist
import tempfile
import subprocess
import shlex
import scipy.io
import socket
import glob
import pandas as pd
import shutil
import vislab.image


def caffe(image_ids, image_filenames, layer='fc6', network='alexnet'):
    import caffe.imagenet

    networks = {
        'alexnet': {
            'model_def_file': (
                str(vislab.config['paths']['caffe'] +
                    '/examples/imagenet/imagenet_deploy.prototxt')
            ),
            'pretrained_model': (
                str(vislab.config['paths']['caffe'] +
                    '/examples/imagenet/caffe_reference_imagenet_model')
            )
        }
    }
    if network not in networks:
        raise ValueError('Only networks supported: {}'.format(networks.keys()))

    # Initialize the network (takes ~1 s)
    net = caffe.imagenet.ImageNetClassifier(**networks[network])
    net.caffenet.set_phase_test()
    net.caffenet.set_mode_cpu()

    if layer not in net.caffenet.blobs.keys():
        raise ValueError('Only layers supported for this network: {}'.format(
            net.caffenet.blobs.keys()))

    good_image_ids = []
    feats = []
    for image_id, image_filename in zip(image_ids, image_filenames):
        try:
            # First, run the network fully forward by calling predict.
            # Then, for whatever blob we want, max across image crops.
            net.predict(image_filename)
            feats.append(net.caffenet.blobs[layer].data.max(0).flatten())
            good_image_ids.append(image_id)
        except:
            continue
    return good_image_ids, feats


def size(image_ids, image_filenames):
    """
    Simply return the (h, w, area, aspect_ratio, has_color) of each image.
    """
    good_image_ids = []
    feats = []
    for image_id, filename in zip(image_ids, image_filenames):
        try:
            image = vislab.dataset.get_image_for_filename(filename)
            has_color = 1 if image.ndim > 2 else 0
            h, w = image.shape[:2]
            feat = np.array((h, w, h * w, float(h) / w, has_color))
            good_image_ids.append(image_id)
            feats.append(feat)
        except:
            continue
    return good_image_ids, feats


def gist(image_ids, image_filenames, max_size=256):
    good_image_ids = []
    feats = []
    for image_id, filename in zip(image_ids, image_filenames):
        try:
            # TODO: resize image to a smaller size? like 128?
            img = vislab.dataset.get_image_for_filename(filename)
            assert(img.dtype == np.uint8)

            if img.ndim == 2:
                img = np.tile(img[:, :, np.newaxis], (1, 1, 3))
            h, w = img.shape[:2]

            mode = 'RGBA'
            rimg = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
            if len(rimg[0]) == 3:
                rimg = np.c_[rimg, 255 * np.ones((len(rimg), 1), np.uint8)]

            im = Image.frombuffer(
                mode, (w, h), rimg.tostring(), 'raw', mode, 0, 1)
            im.thumbnail((max_size, max_size), Image.ANTIALIAS)
            feat = leargist.color_gist(im)

            good_image_ids.append(image_id)
            feats.append(feat)
        except:
            continue
    return image_ids, feats


def lab_hist(image_ids, image_filenames):
    """
    Standard feature as described in [1].
    A histogram in L*a*b* space, having 4, 14, and 14 bins in each dimension
    respectively, for a total of 784 dimensions.

    [1] Palermo, F., Hays, J., & Efros, A. A. (2012).
        Dating Historical Color Images. In ECCV.
    """
    f, output_filename = tempfile.mkstemp()
    image_filenames_cell = '{' + ','.join(
        "'{}'".format(x) for x in image_filenames) + '}'
    matlab = "addpath('matlab/lab_histogram'); lab_hist({}, '{}')".format(
        image_filenames_cell, output_filename)
    matlab_cmd = "matlab -nojvm -r \"try; {}; catch; exit; end; exit\"".format(
        matlab)
    print(matlab_cmd)

    pid = subprocess.Popen(
        shlex.split(matlab_cmd), stdout=open('/dev/null', 'w'))
    retcode = pid.wait()
    if retcode != 0:
        raise Exception("Matlab script did not exit successfully!")

    # Read features
    feats = [x for x in np.loadtxt(output_filename)]
    os.remove(output_filename)

    assert(len(feats) == len(image_ids))
    return image_ids, feats


def gbvs_saliency(image_ids, image_filenames):
    f, output_filename = tempfile.mkstemp()
    output_filename += '.mat'

    image_ids_cell = '{' + ','.join(
        "'{}'".format(x) for x in image_filenames) + '}'
    matlab_script = "get_maps({}, '{}')".format(
        image_ids_cell, output_filename)
    matlab_cmd = "matlab -nojvm -r \"try; {}; catch; exit; end; exit\"".format(
        matlab_script)
    print(matlab_cmd)

    pid = subprocess.Popen(
        shlex.split(matlab_cmd),
        cwd=os.path.expanduser('~/work/vislab/matlab/gbvs'))
    retcode = pid.wait()
    if retcode != 0:
        raise Exception("Matlab script did not exit successfully!")

    # Read features
    try:
        maps = scipy.io.loadmat(output_filename)['maps']
        feats = [x for x in maps]
        os.remove(output_filename)
    except Exception as e:
        raise Exception('Exception {} occured on {}'.format(
            e, socket.gethostname()))

    print("Successfully computed {} features".format(len(feats)))

    assert(len(feats) == len(image_ids))
    return image_ids, feats


def mc_bit(image_ids, image_filenames):
    """
    Compute the mc_bit feature provided by the vlg_extractor package,
    which should be installed in ext/.
    """
    input_dirname = os.path.dirname(image_filenames[0])
    image_filenames = [
        os.path.relpath(fname, input_dirname) for fname in image_filenames]
    f, list_filename = tempfile.mkstemp()
    with open(list_filename, 'w') as f:
        f.write('\n'.join(image_filenames) + '\n')

    output_dirname = tempfile.mkdtemp()
    cmd = './vlg_extractor.sh'
    cmd += ' --parameters-dir={} --extract_mc_bit=ASCII {} {} {}'.format(
        'data/picodes_data', list_filename, input_dirname, output_dirname)
    print(cmd)

    try:
        print("Starting {}".format(cmd))
        p = subprocess.Popen(
            shlex.split(cmd),
            cwd=os.path.expanduser(vislab.config['paths']['vlg_extractor'])
        )
        p.wait()
    except Exception as e:
        print(e)
        raise Exception("Something went wrong with running vlg_extractor")

    image_ids = []
    feats = []
    for filename in glob.glob(output_dirname + '/*_mc_bit.ascii'):
        id_ = os.path.basename(filename.replace('_mc_bit.ascii', ''))
        image_ids.append(id_)
        feats.append(pd.read_csv(filename).values.flatten().astype(bool))

    if p.returncode != 0 or len(feats) == 0:
        raise Exception("Something went wrong with running vlg 2")

    os.remove(list_filename)
    shutil.rmtree(output_dirname)

    return image_ids, feats
