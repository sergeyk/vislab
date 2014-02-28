"""
This file, vislab/ui/app.py, is the updated data and results viewing ui.
The older code for the same tasks is in vislab/app.py
We start with the Pinterest data here.

TODO
- Switch to getting data from a mongo database instead of loading
DataFrames here.
- deal with getting many of a single user's pins at once: maybe only
display at most N pins per user.
- display info about image under it
"""
import os
import flask
import pandas as pd

app = flask.Flask(__name__)

pins_df = pd.read_hdf(os.path.expanduser(
    '~/work/vislab/data/shared/pins_df_feb27.h5'), 'df')
query_names = [
    _[6:] for _ in pins_df.columns.tolist()
    if _.startswith('query_')
]


@app.route('/')
def index():
     return flask.redirect(flask.url_for('data', style_name='all', page=1))


@app.route('/data/<style_name>/<int:page>')
def data(style_name, page):
    results_per_page = 60

    # Filter on style.
    df = pins_df
    if style_name != 'all':
        df = pins_df[pins_df['query_{}'.format(style_name)]]

    # Paginate
    num_pages = df.shape[0] / results_per_page
    start = page * results_per_page
    df = df.iloc[start:min(df.shape[0], start + results_per_page)]

    # Set filter options
    select_options = [
        ('query', ['all'] + query_names, style_name),
        ('page', range(1, num_pages), page)
    ]

    # Fetch images and render.
    images = [_[1].to_dict() for _ in df.iterrows()]
    return flask.render_template(
        'data.html', images=images, select_options=select_options
    )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
