import sys
import re
import pandas as pd
import signal


def vw_filter(df_filename, input_filename):
    """
    Load DataFrame, and read lines from input_filename or stdin.
    If the first word of a line matches to an index in the DataFrame,
    append the label and importance information to the line, and let it
    through in the VW format.

    The DataFrame can have non-unique indices.

    Input format:
    ^ id<image_id> |<feature_name> <feature_id>:<feature_val> |...
    e.g. " id12402 |sift 0:.12 1:.13 id12402 |gist 0:.12 1:.13"

    Parameters
    ----------
    df_filename: string
        Filename of the DataFrame containing label and importance info,
        stored in HDF or pickle format.
    """
    try:
        df = pd.read_hdf(df_filename, 'df')
    except:
        df = pd.read_pickle(df_filename)

    r = re.compile(' id([^ ]+)')
    with input_filename as f:
        for line in f:
            ids = r.findall(line)

            # Make sure there is an id and all ids are the same.
            assert(len(ids) > 0)
            assert([id_ == ids[0] for id_ in ids])

            if ids[0] in df.index:
                clean_line = r.sub('', line)

                # NOTE: getting the column before indexing is important:
                # preserves dtype of the label.
                labels = df['label'].ix[ids[0]]
                importances = df['importance'].ix[ids[0]]

                # If the df has non-unique indices, the above will return
                # iterables.
                if not hasattr(labels, '__iter__'):
                    labels = [labels]
                    importances = [importances]

                for label, importance in zip(labels, importances):
                    whole_label = '{} {:.6f} {}'.format(
                        label, importance, ids[0])
                    sys.stdout.write(whole_label + clean_line[1:])


if __name__ == '__main__':
    """
    Can stream feature data to this script, or have it read from a file.

    usage: cat <input_filename> | vw_filter.py <df_filename>
    or:    vw_filter.py <df_filename> <input_filename>
    """
    # NOTE: this prevents IOError: [Errno 32] Broken pipe
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    assert(len(sys.argv) in [2, 3])
    df_filename = sys.argv[1]
    input_filename = sys.stdin
    if len(sys.argv) > 2:
        input_filename = open(sys.argv[2])

    vw_filter(df_filename, input_filename)
