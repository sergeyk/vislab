"""
Train and test Vowpal Wabbit classifier or regressor on data.
Training includes cross-validation over parameters.
"""
import socket
import cPickle
import bson
import vislab
import vislab.results
import vislab.vw3


def train_and_test(
        collection_name, dataset, feature_names,
        force=False, num_workers=6,
        num_passes=[10], loss=['logistic'], l1=[0], l2=[0],
        quadratic='', bit_precision=18, verbose=False):
    """
    Train and test using VW with the given features and quadratic
    expansion.
    Features are assumed to be stored in VW format in canonical location.
    Cross-validates over all combinations of given parameter choices.

    Parameters
    ----------
    collection_name: string
        Name of the MongoDB 'predict' database collection that contains
        the prediction results.
    dataset: dict
        Contains name information and DataFrames for train, val,
        and test splits.
    feature_names: list of string
        Features to use.
    force: boolean [False]
    num_workers: int [4]
        VW parameter tuning will run in parallel with this many workers,
        on the same machine and reading from the same cache file.
    num_passes: list of int
    loss: list of string [['logistic']]
        Acceptable choices are [
            'hinge', 'squared', 'logistic', 'quantile'
        ]
    l1_weight: list of float [[0]]
    l2_weight: list of float [[0]]
    quadratic: string ['']
        If a non-empty string is given, it must be a sequence of single
        letter corresponding to the namespaces that will be crossed, or
        the word 'all'.
    verbose: boolean [False]
    """
    print("{} running VW on {} for {}".format(
        socket.gethostname(), dataset['name'], dataset['task']))
    print("Using {}, quadratic: {}".format(feature_names, quadratic))

    # To check for existing record of the final score, we construct a
    # query document.
    document = {
        'features': feature_names,
        'task': dataset['task'],
        'quadratic': quadratic
    }
    # The salient parts include the type of data: 'rating', 'style', etc.
    document.update(dataset['salient_parts'])

    # The results are stored in a Mongo database called 'predict'.
    client = vislab.util.get_mongodb_client()
    collection = client['predict'][collection_name]
    if not force:
        result = collection.find_one(document)
        if result is not None:
            print("Already classified this, not doing anything.")
            print(document)
            print("(Score was {:.3f})".format(result['score_test']))
            return

    dirname = vislab.util.makedirs('{}/{}'.format(
        vislab.config['paths']['predict'], dataset['name']))

    # Run VW.
    feat_dirname = \
        vislab.config['paths']['feats'] + '/' + dataset['dataset_name']
    vw = vislab.vw3.VW(
        dirname, num_workers, bit_precision, num_passes, loss, l1, l2)
    pred_df, test_score, val_score, train_score = vw.fit_and_predict(
        dataset, feature_names, feat_dirname, force)

    original_document = document.copy()
    document.update({
        'score_test': test_score,
        'score_val': val_score,
        'pred_df': bson.Binary(cPickle.dumps(pred_df, protocol=2))
    })
    collection.update(original_document, document, upsert=True)
    print("Final score: {:.3f}".format(test_score))
