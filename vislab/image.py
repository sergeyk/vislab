import skimage


def get_image_for_filename(image_filename):
    if image_filename is not None:
        image = skimage.io.imread(image_filename)
        return image
    else:
        return None
