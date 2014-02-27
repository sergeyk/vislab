import os
import shutil
import time
import h5py
import scipy.sparse
import tempfile
import subprocess
import shlex


def dsift_llc(image_filenames, image_ids):
    """
    For each image_id in the list, compute the Dense SIFT feature, LLC coded
    with a 10K-dimensional codebook (obtained by subsampling a 100K-dimensional
    codebook.

    Store the result in the mongo database.

    Uses Stanford Vision Group's hedging-1.0 release code in a matlab call.
    The setup cost of the matlab call is the reason why this function takes
    a list and not a single image_id.

    Parallelism notes:
    - Images are converted to PGM format, and this is done roughly in parallel
    for all images (staggered by the time it takes to download the image if not
    already cached).
    - The matlab script does some matrix computations that take up the
    usual specified number of threads.

    Parameters
    ----------
    image_ids: list

    Returns
    -------
    ids: list of strings
    feats: list of ndarrays
    """
    t = time.time()

    # Set up constants for the featurization code.
    max_size = 500
    knn = 5
    step = 4
    codebook_size = 1000
    feature_size = codebook_size * 10  # 10 bins of the [1x1, 3x3] pyramid
    codebook = 'codebook_{}.mat'.format(codebook_size)
    # The smaller codebook is generated in matlab with:
    # load('codebook_10000.mat')
    # ind = randperm(10000); codebook = codebook(:, ind(1:1000))
    # clear ind; save 'codebook_1000.mat'

    # Fetch and convert images to .pgm format and store in a temp directory.
    temp_dirname = tempfile.mkdtemp()
    pids = []
    temp_filenames = []
    for image_filename, image_id in zip(image_filenames, image_ids):
        # Convert the image to a PGM file and save into the temp directory.
        image_basename = os.path.basename(image_filename)
        temp_filenames.append(temp_dirname + '/{}.pgm'.format(image_basename))
        cmd = "convert -resize {0}x{0}\> -depth 8 {1} {2}".format(
            max_size, image_filename, temp_filenames[-1])
        pids.append(subprocess.Popen(
            shlex.split(cmd), stdout=open(os.devnull, 'wb')))

    # Wait for all conversions to finish
    for pid in pids:
        pid.wait()

    # Check that all images were successfully converted
    for fname in temp_filenames:
        if not os.path.exists(fname):
            raise Exception("Failed to convert at least one image.")

    # Compute features for all files in the temp directory.
    output_fname = temp_dirname + '/feat.mat'
    matlab_script = """
dirname='{}'; knn={}; step={}; codebook='{}'; output='{}'; im2llc_script""".format(
        temp_dirname, knn, step, codebook, output_fname)

    matlab_cmd = """
matlab -nojvm -r "try; {}; catch; disp(lasterr); exit; end; exit""".format(
        matlab_script)
    print(matlab_cmd)

    pid = subprocess.Popen(
        shlex.split(matlab_cmd), cwd='matlab/hedging-1.0',
        stdout=open('/dev/null', 'w'))
    retcode = pid.wait()
    if retcode != 0:
        raise Exception("Matlab script did not exit successfully!")

    # Load the features from the .mat file.
    with h5py.File(output_fname, 'r') as f:
        ids = [str(u''.join(unichr(c) for c in f[obj_ref])) for obj_ref in f['ids'][0]]
        feats = scipy.sparse.csc_matrix((f['betas']['data'], f['betas']['ir'], f['betas']['jc']), shape=[feature_size, len(ids)])
    feats = [feats[:, i] for i in range(feats.shape[1])]

    # Delete the temporary directory.
    shutil.rmtree(temp_dirname)

    print("dsift_llc: {} new images processed in {:.3f} s".format(
        len(ids), time.time() - t))

    return ids, feats
