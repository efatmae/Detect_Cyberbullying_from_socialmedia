import argparse
import os, sys
import urllib
import json
import pandas as pd
from baselines import one_hot, empirical_dist
from deep_learning import make_mlp, DenseTransformer, make_lstm,SequenceTransformer, make_conv_lstm
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from keras.wrappers.scikit_learn import KerasClassifier
from serialization import save_pipeline, load_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
import tensorflow as tf
import keras
from datetime import datetime
"""
USAGE EXAMPLE:
python get_prod_models.py --task attack --model_dir ../../app/models
python wiki-detox/src/modeling/get_prod_models.py --task recipient_attack --model_dir ../../app/models
python get_prod_models.py --task aggression --model_dir ../../app/models
python get_prod_models.py --task toxicity --model_dir ../../app/models

"""
start=datetime.now()
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
# Figshare URLs for downloading training data
ATTACK_ANNOTATED_COMMENTS_URL = 'https://ndownloader.figshare.com/files/7554634'
ATTACK_ANNOTATIONS_URL = 'https://ndownloader.figshare.com/files/7554637'
AGGRESSION_ANNOTATED_COMMENTS_URL = 'https://ndownloader.figshare.com/files/7038038'
AGGRESSION_ANNOTATIONS_URL = 'https://ndownloader.figshare.com/files/7383748'
TOXICITY_ANNOTATED_COMMENTS_URL = 'https://ndownloader.figshare.com/files/7394542'
TOXICITY_ANNOTATIONS_URL = 'https://ndownloader.figshare.com/files/7394539'
# CSV of optimal  hyper-parameters for each model architecture
CV_RESULTS = 'cv_results.csv'
Project_Path = '/home/fatma/Dropbox/Tensorbook/Ex_Machina_Replication/wiki-detox-master'


def download_file(url, fname):
    """
    Helper function for downloading a file to disk
    """
    urllib.request.urlretrieve(url, fname)

def download_training_data(data_dir, task):

    """
    Downloads comments and labels for task
    """

    COMMENTS_FILE = "%s_annotated_comments.tsv" % task
    LABELS_FILE = "%s_annotations.tsv" % task

    if task == "attack":
        download_file(ATTACK_ANNOTATED_COMMENTS_URL,
                      os.path.join(data_dir, COMMENTS_FILE))
        download_file(ATTACK_ANNOTATIONS_URL, os.path.join(data_dir,
                      LABELS_FILE))
    elif task == "recipient_attack":
        download_file(ATTACK_ANNOTATED_COMMENTS_URL,
                      os.path.join(data_dir, COMMENTS_FILE))
        download_file(ATTACK_ANNOTATIONS_URL, os.path.join(data_dir,
                      LABELS_FILE))
    elif task == "aggression":
        download_file(AGGRESSION_ANNOTATED_COMMENTS_URL,
                      os.path.join(data_dir, COMMENTS_FILE))
        download_file(AGGRESSION_ANNOTATIONS_URL,
                      os.path.join(data_dir, LABELS_FILE))
    elif task == "toxicity":
        download_file(TOXICITY_ANNOTATED_COMMENTS_URL,
                      os.path.join(data_dir, COMMENTS_FILE))
        download_file(TOXICITY_ANNOTATIONS_URL,
                      os.path.join(data_dir, LABELS_FILE))
    else:
        print("No training data for task: ", task)

def parse_training_data(data_dir, task):

    """
    Computes labels from annotations and aligns comments and labels for training
    """

    COMMENTS_FILE = "%s_annotated_comments.tsv" % task
    LABELS_FILE = "%s_annotations.tsv" % task

    print(os.path.join(Project_Path, data_dir, COMMENTS_FILE))
    comments = pd.read_csv(os.path.join(Project_Path, data_dir, COMMENTS_FILE), sep = ',', index_col = 0)
    #print(comments.columns)
    # remove special newline and tab tokens
    comments['Text'] = comments['Text'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
    comments['Text'] = comments['Text'].apply(lambda x: x.replace("TAB_TOKEN", " "))

    annotations = pd.read_csv(os.path.join(Project_Path, data_dir, LABELS_FILE),  sep = ',', index_col = 0)
    labels = empirical_dist(annotations['oh_label'])
    X = comments.sort_index()['Text'].values
    y = labels.sort_index().values

    assert(X.shape[0] == y.shape[0])
    return X, y

def convert_one_target_colum_into_two(target_col):
        y_0 = [1 if i == 0 else 0 for i in target_col]
        y_1 = [1 if i == 1 else 0 for i in target_col]
        y_binary = np.column_stack((y_0, y_1))
        return y_binary

def read_data(data_dir,task, label_type):

    data_file_name = '%s_parsed_dataset.csv' % task
    comments = pd.read_csv(os.path.join(Project_Path, data_dir, data_file_name), index_col=0)
    # print(comments.columns)
    # remove special newline and tab tokens
    comments['Text'] = comments['Text'].apply(lambda x: str(x).replace("NEWLINE_TOKEN", " "))
    comments['Text'] = comments['Text'].apply(lambda x: str(x).replace("TAB_TOKEN", " "))

    if label_type == "oh":
        labels = comments['oh_label']
        print(comments['oh_label'].unique())
        labels = convert_one_target_colum_into_two(labels)

    else:
        labels = np.column_stack((comments['ed_label_0'],comments['ed_label_1']))

    X = comments['Text'].values
    y = labels

    assert (X.shape[0] == y.shape[0])
    return X, y

def train_model(X, y, model_type, ngram_type, label_type):
    """
    Trains a model with the specified architecture. Note that the
    classifier is a Sklearn model when setting label_type == 'oh'
    and model_type == 'linear'. Otherwise the classifier is a
    Keras model. The distinction is important for serialization.
    """
    assert(label_type in ['oh', 'ed'])
    assert(model_type in ['linear', 'mlp', 'lstm',"conv_lstm"])
    assert(ngram_type in ['word', 'char'])
    # tensorflow models aren't fork safe, which means they can't be served via uwsgi
    # as work around, we can serve a pure sklearn model
    # we should be able to find another fix

    if label_type == 'oh' and model_type == 'linear':

        y = np.argmax(y, axis = 1)
        print("linear + OH", y)
        clf = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', LogisticRegression()),
        ])

        params = {
            'vect__max_features': 10000,
            'vect__ngram_range': (1,2),
            'vect__analyzer' : ngram_type,
            'tfidf__sublinear_tf' : True,
            'tfidf__norm' :'l2',
            'clf__C' : 10,
        }

    elif model_type == "mlp":
        if label_type == 'oh':
            y = one_hot(y)
        clf = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('to_dense', DenseTransformer()),
            ('clf', KerasClassifier(build_fn=make_mlp, output_dim = y.shape[1], verbose=False)),
        ])

        cv_results = pd.read_csv('cv_results.csv')
        query = "model_type == '%s' and ngram_type == '%s' and label_type == '%s'" % (model_type, ngram_type, label_type)
        params = cv_results.query(query)['best_params'].iloc[0]
        params = json.loads(params)


    elif model_type == "lstm":
        if label_type == 'oh':
            y = one_hot(y)
        clf = Pipeline([
            ('seq', SequenceTransformer()),
            ('clf', KerasClassifier(build_fn=make_lstm, output_dim = y.shape[1], verbose=False)),
        ])
        #cv_results = pd.read_csv('cv_results.csv')
        #query = "model_type == '%s' and ngram_type == '%s' and label_type == '%s'" % (model_type, ngram_type, label_type)
        #params = cv_results.query(query)['best_params'].iloc[0]
        params = {"seq__max_features" : 10000,
                  "seq__analyzer" : ngram_type,
                "clf__output_dim" : 2,
                "clf__max_len" : 100,
                "clf__embedding_size" : 64,
                "clf__lstm_output_size" : 128,
                "clf__dropout" : 0.25}
    else:
        if label_type == 'oh':
            y = one_hot(y)
        clf = Pipeline([
            ('seq', SequenceTransformer()),
            ('clf', KerasClassifier(build_fn=make_conv_lstm, output_dim=y.shape[1], verbose=False)),
        ])

        # cv_results = pd.read_csv('cv_results.csv')
        # query = "model_type == '%s' and ngram_type == '%s' and label_type == '%s'" % (model_type, ngram_type, label_type)
        # params = cv_results.query(query)['best_params'].iloc[0]
        params = {"seq__max_features" : 10000,
                "clf__output_dim" : 2,
                "clf__max_len" : 100,
                "clf__embedding_size" : 64,
                "clf__filter_length" : 3,
                "clf__nb_filter" : 64,
                "clf__pool_length" : 4,
                "clf__lstm_output_size" : 128,
                "clf__dropout" : 0.25}
    print("parameters", params)
    return clf.set_params(**params).fit(X,y)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--skip_download',   default = 'true',   help ='do not download data if "true"')
    parser.add_argument('--data_dir',   default = 'Fatma_dataset',   help ='directory for saving training data')
    parser.add_argument('--model_dir',  default = 'tmp/models',   help ='directory for saving model' )
    parser.add_argument('--task',       default = 'Twitter_racism', help = 'either attack, recipient_attack, aggression, toxicity, Twitter-racism. Twitter_sexism, kaggle or YouTube')
    parser.add_argument('--model_type', default = 'linear', help = 'either linear,mlp, lstm or conv_lstm')
    parser.add_argument('--ngram_type', default = 'char',   help = 'either word or char')
    parser.add_argument('--label_type', default = 'oh',    help = 'either oh or ed')
    args = vars(parser.parse_args())

    if args['skip_download'] != 'true':
        print("Downloading Data")
        download_training_data(args['data_dir'], args['task'])
    #print("Parsing Data")
    #X, y = parse_training_data(args['data_dir'], args['task'])
    print("Read Data")
    X, y = read_data(args['data_dir'], args['task'],args['label_type'])
    #split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    print("Training Model")
    print(X_train.shape)
    clf = train_model(X_train, y_train, args['model_type'], args['ngram_type'], args['label_type'])
    print(clf)
    print("Saving Model")
    clf_name = "Fatma_training_set_%s_%s_%s_%s" % (args['task'], args['model_type'], args['ngram_type'], args['label_type'])
    print(clf_name)
    save_pipeline(clf, os.path.join(Project_Path, args['model_dir']), clf_name)
    print("Reloading Model")
    clf = load_pipeline(os.path.join(Project_Path, args['model_dir']), clf_name)

    y_true = [True if i[0] == 0 else False for i in y_test]
    y_prediction = clf.predict_proba(X_test)
        #print(np.unique(y_true))

    print(roc_auc_score(y_true, y_prediction[:, 1]))
    #print(clf.predict_proba(['fuck']))

    y_true2 = [1 if i[0] == 0 else 0 for i in y_test]
    y_prediction2 = clf.predict(X_test)
    print(roc_auc_score(y_true2, y_prediction2))

    results_df = pd.DataFrame({"y_true_bool":y_true,
                               "y_predict_prob_0":y_prediction[:, 0],"y_predict_prob_1":y_prediction[:, 1],
                               "y_true_binary":y_true2,"y_prediction":y_prediction2})
    results_df.to_csv(os.path.join(Project_Path, args["data_dir"],'results', args['model_type'] + '_' + args['ngram_type']+'_'+args['label_type']+'_'+args['task'] + "_" + "prediction_results.csv"))
    with open(os.path.join(Project_Path, args["data_dir"],'results',"timing.txt"),"a")as f:
        f.write(args['model_type'] + '_' + args['ngram_type']+'_'+args['label_type']+'_'+args['task'] + "_" + "processing_time " + str(datetime.now() - start)+ '\n')
        f.close()
