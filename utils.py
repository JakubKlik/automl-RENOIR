import numpy as np
import pandas as pd
from scipy.io import arff

def load_data(stream_name):
    data, meta = arff.loadarff("datasets/%s.arff" % stream_name)
    df = pd.DataFrame(data)
    features = df.iloc[:, 0:-1].values.astype(float)
    labels = df.iloc[:, -1].values.astype(str)
    labels[labels=="negative"] = 0
    labels[labels=="positive"] = 1
    labels = labels.astype(int)
    classes = np.unique(labels)
    return features, labels, classes
