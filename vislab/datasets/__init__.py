import ava
import flickr
import wikipaintings
import pascal
import behance

DATASETS = {
    'ava': {
        'fn': ava.get_ava_df
    },
    'ava_style': {
        'fn': ava.get_style_df
    },
    'flickr': {
        'fn': flickr.load_flickr_df
    },
    'wikipaintings': {
        'fn': wikipaintings.get_style_df
    },
    'wikipaintings_artist': {
        'fn': wikipaintings.get_artist_df
    },
    'pascal': {
        'fn': pascal.get_clf_df
    },
    'behance_photo': {
        'fn': behance.get_photo_df
    }
}
