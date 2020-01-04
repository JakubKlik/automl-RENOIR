from datetime import datetime

import sklearn
import autosklearn
import autosklearn.classification

from sklearn.datasets import make_blobs, make_classification
from sklearn.model_selection import train_test_split

from classifiers import BernoulliNB_S
from classifiers import DecisionTree_S
from classifiers import GaussianNB_S
from classifiers import KNearestNeighborsClassifier_S
from classifiers import LibLinear_SVC_S
from classifiers import LibSVM_SVC_S
from classifiers import RandomForest_S
from classifiers import SGD_S

from utils import load_data

import os
import warnings
warnings.simplefilter("ignore")

autosklearn.pipeline.components.classification.add_classifier(BernoulliNB_S)
autosklearn.pipeline.components.classification.add_classifier(DecisionTree_S)
autosklearn.pipeline.components.classification.add_classifier(GaussianNB_S)
autosklearn.pipeline.components.classification.add_classifier(KNearestNeighborsClassifier_S)
autosklearn.pipeline.components.classification.add_classifier(LibLinear_SVC_S)
autosklearn.pipeline.components.classification.add_classifier(LibSVM_SVC_S)
autosklearn.pipeline.components.classification.add_classifier(RandomForest_S)
autosklearn.pipeline.components.classification.add_classifier(SGD_S)

datasets = [
"abalone-17_vs_7-8-9-10",
"page-blocks0",
"yeast-0-2-5-6_vs_3-7-8-9",
"kc1",
"vowel0",
"pima"
]
for dataset in datasets:

    X, y, c = load_data(dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=True, random_state=1111)


    # now = datetime.now()
    # name = now.strftime("%d-%m-%Y_%H:%M:%S")
    name = dataset

    all_run_time = 60*60
    run_time = int(all_run_time*0.05)
    # run_time = 15

    if not os.path.exists("./logs/exp3_%s" % name):
        os.makedirs("./logs/exp3_%s" % name)

    clfs = [
    "BernoulliNB_S",
    "DecisionTree_S",
    "GaussianNB_S",
    "KNearestNeighborsClassifier_S",
    "LibLinear_SVC_S",
    "LibSVM_SVC_S",
    "RandomForest_S",
    "SGD_S"
    ]

    aclf = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=all_run_time,
        per_run_time_limit=run_time,
        ml_memory_limit=4000,
        ensemble_size=10,
        ensemble_nbest=10,
        ensemble_memory_limit=4000,
        include_estimators=clfs,
        tmp_folder="./logs/exp3_%s/tmp_sampling" % name,
        delete_tmp_folder_after_terminate=False,
        exclude_preprocessors=['balancing','pca'],
        resampling_strategy='cv',
    	resampling_strategy_arguments={'folds': 5},
    )
    aclf.fit(X_train.copy(), y_train.copy(), dataset_name=dataset, metric=autosklearn.metrics.f1)

    aclf.refit(X_train.copy(), y_train.copy())

    y_pred = aclf.predict(X_test)

    with open("./logs/exp3_%s/logfile_sampling.txt" % name, 'at') as logfile:
        print("--------------------------------------------------------------------------------------------------------------------", file=logfile)
        print("---  METRICS  ------------------------------------------------------------------------------------------------------", file=logfile)
        print("--------------------------------------------------------------------------------------------------------------------", file=logfile)
        print("ACC: %0.5f" % autosklearn.metrics.accuracy(y_pred, y_test), file=logfile)
        print("F-1: %0.5f" % autosklearn.metrics.f1(y_pred, y_test), file=logfile)
        print("PRC: %0.5f" % autosklearn.metrics.precision(y_pred, y_test), file=logfile)
        print("REC: %0.5f" % autosklearn.metrics.recall(y_pred, y_test), file=logfile)
        print("BAC: %0.5f" % autosklearn.metrics.balanced_accuracy(y_pred, y_test), file=logfile)
        print("--------------------------------------------------------------------------------------------------------------------", file=logfile)
        print("---  MODELS  -------------------------------------------------------------------------------------------------------", file=logfile)
        print("--------------------------------------------------------------------------------------------------------------------", file=logfile)
        print(aclf.show_models(), file=logfile)
        print("--------------------------------------------------------------------------------------------------------------------", file=logfile)
        print("---  CV RESULTS  ---------------------------------------------------------------------------------------------------", file=logfile)
        print("--------------------------------------------------------------------------------------------------------------------", file=logfile)
        print(aclf.cv_results_, file=logfile)
        print("--------------------------------------------------------------------------------------------------------------------", file=logfile)
        print("---  STATS  --------------------------------------------------------------------------------------------------------", file=logfile)
        print("--------------------------------------------------------------------------------------------------------------------", file=logfile)
        print(aclf.sprint_statistics(), file=logfile)



    clfs = [
    # "adaboost",
    "bernoulli_nb",
    "decision_tree",
    # "extra_trees",
    "gaussian_nb",
    # "gradient_boosting",
    "k_nearest_neighbors",
    # "lda",
    "liblinear_svc",
    "libsvm_svc",
    # "multinomial_nb",
    # "passive_aggressive",
    # "qda",
    "random_forest",
    "sgd",
    ]


    aclf = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=all_run_time,
        per_run_time_limit=run_time,
        ml_memory_limit=4000,
        ensemble_size=10,
        ensemble_nbest=10,
        ensemble_memory_limit=4000,
        include_estimators=clfs,
        tmp_folder="./logs/exp3_%s/tmp_normal" % name,
        delete_tmp_folder_after_terminate=False,
        exclude_preprocessors=['balancing'],
        resampling_strategy='cv',
    	resampling_strategy_arguments={'folds': 5},
    )
    aclf.fit(X_train.copy(), y_train.copy(), dataset_name=dataset, metric=autosklearn.metrics.f1)

    aclf.refit(X_train.copy(), y_train.copy())

    y_pred = aclf.predict(X_test)

    with open("./logs/exp3_%s/logfile_normal.txt" % name, 'at') as logfile:
        print("--------------------------------------------------------------------------------------------------------------------", file=logfile)
        print("---  METRICS  ------------------------------------------------------------------------------------------------------", file=logfile)
        print("--------------------------------------------------------------------------------------------------------------------", file=logfile)
        print("ACC: %0.5f" % autosklearn.metrics.accuracy(y_pred, y_test), file=logfile)
        print("F-1: %0.5f" % autosklearn.metrics.f1(y_pred, y_test), file=logfile)
        print("PRC: %0.5f" % autosklearn.metrics.precision(y_pred, y_test), file=logfile)
        print("REC: %0.5f" % autosklearn.metrics.recall(y_pred, y_test), file=logfile)
        print("BAC: %0.5f" % autosklearn.metrics.balanced_accuracy(y_pred, y_test), file=logfile)
        print("--------------------------------------------------------------------------------------------------------------------", file=logfile)
        print("---  MODELS  -------------------------------------------------------------------------------------------------------", file=logfile)
        print("--------------------------------------------------------------------------------------------------------------------", file=logfile)
        print(aclf.show_models(), file=logfile)
        print("--------------------------------------------------------------------------------------------------------------------", file=logfile)
        print("---  CV RESULTS  ---------------------------------------------------------------------------------------------------", file=logfile)
        print("--------------------------------------------------------------------------------------------------------------------", file=logfile)
        print(aclf.cv_results_, file=logfile)
        print("--------------------------------------------------------------------------------------------------------------------", file=logfile)
        print("---  STATS  --------------------------------------------------------------------------------------------------------", file=logfile)
        print("--------------------------------------------------------------------------------------------------------------------", file=logfile)
        print(aclf.sprint_statistics(), file=logfile)
