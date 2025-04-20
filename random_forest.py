from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score
from score import report_score  # make sure this is implemented
import pandas as pd
import pickle
from CountFeatureGenerator import *
from TfidfFeatureGenerator import *
from SvdFeatureGenerator import *
from Word2VecFeatureGenerator import *
from SentimentFeatureGenerator import *
from score import *
import score

def build_data():
    data = pd.read_csv('./data/train.csv', encoding = 'utf-8')
    used_columns = ['id', 'keyword', 'text', 'target']
    data = data[used_columns].dropna()

    
    train = data.sample(frac=0.8, random_state=2025)
    test = data.loc[~data.index.isin(train.index)]

    data_y = train['target'].values
    generators = [
                  CountFeatureGenerator(),
                  TfidfFeatureGenerator(),
                  SvdFeatureGenerator(),
                  Word2VecFeatureGenerator(),
                  SentimentFeatureGenerator()
                 ]
    features = [f for g in generators for f in g.read('train')]
    features = [f.toarray() if hasattr(f, 'toarray') else f for f in features]  #Check if have toarray method (method of sparse matric but not possessed by numpy array)

    features_fixed = []
    for i, feat in enumerate(features):
        if isinstance(feat, np.ndarray):
            if feat.ndim == 1:
                print(f"🔧 Fixing feature {i} from shape {feat.shape}")
                feat = feat.reshape(-1, 1)  # Convert (4064,) → (4064, 1)
            features_fixed.append(feat)
        else:
            print(f"❌ Feature {i} is not a NumPy array, it's {type(feat)}")

    data_x = (np.hstack(features_fixed))
    print(data_x[0,:])
    print('data_x.shape')
    print(data_x.shape)
    print('data_y.shape')
    print(data_y.shape)
    print('body_ids.shape')
    print(data['id'].values.shape)


    return data_x, data_y, data['id'].values, test[['id', 'keyword', 'text']]


def build_test_data():

    data = pd.read_csv('./data/train.csv', encoding = 'utf-8')
    used_columns = ['id', 'keyword', 'text', 'target']
    data = data[used_columns].dropna()

    
    train = data.sample(frac=0.8, random_state=2025)
    test = data.loc[~data.index.isin(train.index)]

    generators = [
            CountFeatureGenerator(),
            TfidfFeatureGenerator(),
            SvdFeatureGenerator(),
            Word2VecFeatureGenerator(),
            SentimentFeatureGenerator()
                 ]

    features = [f for g in generators for f in g.read("test")]
    features = [f.toarray() if hasattr(f, 'toarray') else f for f in features]  #Check if have toarray method (method of sparse matric but not possessed by numpy array)

    features_fixed = []
    for i, feat in enumerate(features):
        if isinstance(feat, np.ndarray):
            if feat.ndim == 1:
                print(f"🔧 Fixing feature {i} from shape {feat.shape}")
                feat = feat.reshape(-1, 1)  # Convert (4064,) → (4064, 1)
            features_fixed.append(feat)
        else:
            print(f"❌ Feature {i} is not a NumPy array, it's {type(feat)}")

    for i, f in enumerate(features_fixed):
        print(f"Feature {i} shape: {f.shape}")
    data_x = np.hstack(features_fixed)
    print(data_x[0,:])
    print('test data_x.shape')
    print(data_x.shape)
    print('test body_ids.shape')
    print(test['id'].values.shape)
                   # pair id
    return data_x, test['id'].values, test['target']

def train():
    