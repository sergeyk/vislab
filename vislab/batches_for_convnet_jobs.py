# aggregate all training images from flickr and wikipaintings,
# format them in the 3x256x256 way, and assign labels in the way
# jeff's scripts expect

import vislab
import cPickle
import vislab.predict
import vislab.datasets


def get_train_df():
    flickr_df = vislab.datasets.flickr.load_flickr_df()
    wp_df = vislab.datasets.wikipaintings.get_style_df()
    wp_df = wp_df.join(vislab.datasets.wikipaintings.get_df())

    flickr_dataset = vislab.predict.get_multiclass_dataset(
        flickr_df, 'flickr',
        'all_style', vislab.datasets.flickr.underscored_style_names)

    wp_styles = [x for x in wp_df.columns if x.startswith('style_')]
    wp_dataset = vislab.predict.get_multiclass_dataset(
        wp_df, 'wikipaintings',
        'all_style', wp_styles)

    flickr_train_df = flickr_dataset['train_df']
    flickr_train_df['image_url'] = flickr_df['image_url']
    flickr_train_df['dataset_name'] = 'flickr'

    wp_train_df = wp_dataset['train_df']
    wp_train_df['image_url'] = wp_df['image_url']
    wp_train_df['dataset_name'] = 'wikipaintings'

    # flickr labels will be first
    wp_train_df['label'] += flickr_train_df['label'].nunique()

    # Make one dataframe, and make the label 0-indexed.
    cols = ['label', 'image_url', 'dataset_name']
    train_df = flickr_train_df[cols].append(wp_train_df[cols])
    train_df['label'] -= 1

    return train_df


if __name__ == '__main__':
    train_df = get_train_df()

    # Submit jobs to queue
    queue_name = 'finetune_convnet'
    redis_conn = vislab.util.get_redis_client()
    for image_id, row in train_df.iterrows():
        redis_conn.rpush(queue_name, cPickle.dumps((image_id, dict(row))))
