---
layout: default
title: Datasets
---

# Datasets

Vislab provides support for several vision datasets out of the box.

## Provided

pascal
: VOC2012, ~10K images tagged with 20 object classes. Multi-label.

ava
: ~250K images with aesthetic ratings

ava_style
: ~20K images from AVA that also have style labels

flickr
: ~50K images with style labels

wikipaintings
: ~100K images with style, genre, artist labels

## Adding your own

Adding your own dataset is super simple.

1. Add the file `vislab/datasets/your_dataset.py` that will contain a function to load a `pandas.DataFrame` with:
- unique string-based index, with name `image_id`
- `image_url` or `image_filename` in columns
- a column for whatever boolean label you care about
2. Modify `DATASETS` in `vislab/dataset.py` to map a name to your new function.

## Dataset for classification

The classifier expects DataFrames with only two columns: 'label' and 'importance'.

The 'label' column can contain:

- real values (regression)
- -1/1 (binary classification)
- or positive ints (multiclass classification).
