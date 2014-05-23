import ava
import flickr
import wikipaintings
import pinterest
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
        'fn': flickr.get_df
    },
    'wikipaintings': {
        'fn': wikipaintings.get_style_df
    },
    'wikipaintings_artist': {
        'fn': wikipaintings.get_artist_df
    },
    'pinterest_80k': {
        'fn': pinterest.get_pins_80k_df
    },
    'pascal': {
        'fn': pascal.get_class_df
    },
    'pascal_mc': {
        'fn': pascal.get_metaclass_df
    },
    'pascal_det': {
        'fn': pascal.get_det_df
    },
    'behance_photo': {
        'fn': behance.get_photo_df
    }
}
