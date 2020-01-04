
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


selector = {
    "ADASYN": ADASYN(),
    "BorderlineSMOTE": BorderlineSMOTE(),
    "KMeansSMOTE": KMeansSMOTE(),
    "RandomOverSampler": RandomOverSampler(),
    "SMOTE": SMOTE(),
    # "SMOTENC": SMOTENC(),
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

print(selector.get("SMOTE"))
