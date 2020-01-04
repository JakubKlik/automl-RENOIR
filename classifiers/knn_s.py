from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter, UniformFloatHyperparameter

import sklearn.metrics
import autosklearn.classification
import autosklearn.pipeline.components.classification
from autosklearn.pipeline.components.base import AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import DENSE, SIGNED_DATA, UNSIGNED_DATA, PREDICTIONS

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

import sklearn.neighbors
import sklearn.multiclass

class KNearestNeighborsClassifier_S(AutoSklearnClassificationAlgorithm):

    def __init__(self,
                 n_neighbors,
                 weights,
                 p,
                 sampling_method,
                 random_state=None):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p
        self.random_state = random_state
        self.sampling_method = sampling_method


    def fit(self, X, y):

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

        estimator = sklearn.neighbors.KNeighborsClassifier(n_neighbors=self.n_neighbors,
                                                   weights=self.weights,
                                                   p=self.p)

        if len(y.shape) == 2 and y.shape[1] > 1:
            self.estimator = sklearn.multiclass.OneVsRestClassifier(estimator, n_jobs=1)
        else:
            self.estimator = estimator

        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict_proba(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'KNN-S',
                'name': 'K-Nearest Neighbor Classification Sampling',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                'input': [DENSE, SIGNED_DATA, UNSIGNED_DATA],
                'output': [PREDICTIONS]
                }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        n_neighbors = UniformIntegerHyperparameter(
            name="n_neighbors", lower=1, upper=100, log=True, default_value=1)
        weights = CategoricalHyperparameter(
            name="weights", choices=["uniform", "distance"], default_value="uniform")
        p = CategoricalHyperparameter(name="p", choices=[1, 2], default_value=2)
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

        cs.add_hyperparameters([n_neighbors,
                                weights,
                                p,
                                sampling_method
                                ])

        return cs
