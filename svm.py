import sys
import dill as pickle
import numpy as np
from itertools import chain
from collections import Counter
from sklearn.model_selection import StratifiedKFold, GroupKFold
import xgboost as xgb
from sklearn import svm
from collections import Counter
from CountFeatureGenerator import *
from TfidfFeatureGenerator import *
from SvdFeatureGenerator import *
from Word2VecFeatureGenerator import *
from SentimentFeatureGenerator import *
from score import *
import score
import joblib
from sklearn.metrics import confusion_matrix


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

    data_x, data_y, body_ids, target_stance = build_data()
    print(data_x, data_y, body_ids, target_stance)
    # read test data
    test_x, body_ids_test, true_y = build_test_data()

    svm_FNC = svm.SVC(kernel='linear', probability=True)
    bst = svm_FNC.fit(data_x, data_y)

    pred_y = bst.predict(test_x)
    print(len(pred_y))

    df_test = pd.DataFrame()
    df_test['pred_y'] = pred_y
    df_test['true_y'] = true_y
    df_test = df_test.dropna()

    print(df_test['pred_y'].value_counts())
    print(df_test['true_y'].value_counts())

    print(score.report_score(df_test['true_y'], df_test['pred_y']))

    predicted = [LABELS[int(a)] for a in pred_y]
    stances = target_stance

    df_output = pd.DataFrame()
    df_output['id'] = body_ids_test
    df_output['pred'] = pred_y
    
    df_output.to_csv('svm_pred_prob.csv', index=False)


if __name__ == '__main__':
    train()


#Name: count, dtype: int64
# -------------------------------------------------
# |               | not_disaster  |   disaster    |
# -------------------------------------------------
# | not_disaster  |      110      |      72       |
# -------------------------------------------------
# |   disaster    |      86       |      24       |
# -------------------------------------------------

# Classification Report:
#                precision    recall  f1-score   support

# not_disaster       0.56      0.60      0.58       182
#     disaster       0.25      0.22      0.23       110

#     accuracy                           0.46       292
#    macro avg       0.41      0.41      0.41       292
# weighted avg       0.44      0.46      0.45       292


#  Binary F1 Score (target: 1): 0.2330
# 0.23300970873786409







