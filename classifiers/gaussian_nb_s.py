import numpy as np

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

from autosklearn.pipeline.components.base import (
    AutoSklearnClassificationAlgorithm,
    IterativeComponent,
)
from autosklearn.pipeline.constants import *


from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import KMeansSMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import SVMSMOTE
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from imblearn.under_sampling import AllKNN
from imblearn.under_sampling import InstanceHardnessThreshold
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks

class GaussianNB_S(IterativeComponent, AutoSklearnClassificationAlgorithm):

    def __init__(self, sampling_method, random_state=None, verbose=0):

        self.random_state = random_state
        self.verbose = int(verbose)
        self.estimator = None
        self.sampling_method = sampling_method

    def iterative_fit(self, X, y, n_iter=1, refit=False):
        import sklearn.naive_bayes

        selector = {
            "ADASYN": ADASYN(),
            "BorderlineSMOTE": BorderlineSMOTE(),
            "KMeansSMOTE": KMeansSMOTE(),
            "RandomOverSampler": RandomOverSampler(),
            "SMOTE": SMOTE(),
            "SVMSMOTE": SVMSMOTE(),
            "ClusterCentroids": ClusterCentroids(),
            "CondensedNearestNeighbour": CondensedNearestNeighbour(),
            "EditedNearestNeighbours": EditedNearestNeighbours(),
            "RepeatedEditedNearestNeighbours": RepeatedEditedNearestNeighbours(),
            "AllKNN": AllKNN(),
            "InstanceHardnessThreshold": InstanceHardnessThreshold(),
            "NearMiss": NearMiss(),
            "NeighbourhoodCleaningRule": NeighbourhoodCleaningRule(),
            "OneSidedSelection": OneSidedSelection(),
            "RandomUnderSampler": RandomUnderSampler(),
            "TomekLinks": TomekLinks()
        }

        self.preprocessor = selector.get(self.sampling_method)

        X,y = self.preprocessor.fit_resample(X,y)

        if refit:
            self.estimator = None

        if self.estimator is None:
            self.n_iter = 0
            self.fully_fit_ = False
            self.estimator = sklearn.naive_bayes.GaussianNB()
            self.classes_ = np.unique(y.astype(int))

        # Fallback for multilabel classification
        if len(y.shape) > 1 and y.shape[1] > 1:
            import sklearn.multiclass
            self.estimator.n_iter = self.n_iter
            self.estimator = sklearn.multiclass.OneVsRestClassifier(
                self.estimator, n_jobs=1)
            self.estimator.fit(X, y)
            self.fully_fit_ = True
        else:
            for iter in range(n_iter):
                start = min(self.n_iter * 1000, y.shape[0])
                stop = min((self.n_iter + 1) * 1000, y.shape[0])
                if X[start:stop].shape[0] == 0:
                    self.fully_fit_ = True
                    break

                self.estimator.partial_fit(X[start:stop], y[start:stop],
                                           self.classes_)
                self.n_iter += 1

                if stop >= len(y):
                    self.fully_fit_ = True
                    break

        return self

    def configuration_fully_fitted(self):
        if self.estimator is None:
            return False
        elif not hasattr(self, 'fully_fit_'):
            return False
        else:
            return self.fully_fit_

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict_proba(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'GaussianNB',
                'name': 'Gaussian Naive Bayes classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                'input': (DENSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        sampling_method = CategoricalHyperparameter(
            name="sampling_method", choices=[
                                             "ADASYN",
                                             "BorderlineSMOTE",
                                             "KMeansSMOTE",
                                             "RandomOverSampler",
                                             "SMOTE",
                                             "SMOTENC",
                                             "SVMSMOTE",
                                             "ClusterCentroids",
                                             "CondensedNearestNeighbour",
                                             "EditedNearestNeighbours",
                                             "RepeatedEditedNearestNeighbours",
                                             "AllKNN",
                                             "InstanceHardnessThreshold",
                                             "NearMiss",
                                             "NeighbourhoodCleaningRule",
                                             "OneSidedSelection",
                                             "RandomUnderSampler",
                                             "TomekLinks"
                                             ], default_value="NearMiss")

        cs.add_hyperparameters([sampling_method])
        return cs
