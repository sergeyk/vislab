"""
This file, vislab/ui/app.py, is the updated data and results viewing ui.
The older code for the same tasks is in vislab/app.py
We start with the Pinterest data here.

TODO
- Switch to getting data from a mongo database instead of loading
DataFrames here.
- deal with getting many of a single user's pins at once: maybe only
display at most N pins per user.
- add drop-down select menu to select which style to view.
- add pagination
"""
import os
import flask
import pandas as pd

app = flask.Flask(__name__)

pins_df = pd.read_hdf(os.path.expanduser(
    '~/work/vislab/data/shared/pins_df_feb27.h5'), 'df')


@app.route('/')
def data():
    df = pins_df.iloc[:50]
    images = [_[1].to_dict() for _ in df.iterrows()]
    return flask.render_template('data.html', images=images)


@app.route('/<style_name>')
def data_for_style(style_name):
    df = pins_df[pins_df['query_{}'.format(style_name)]].iloc[:50]
    images = [_[1].to_dict() for _ in df.iterrows()]
    return flask.render_template('data.html', images=images)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
