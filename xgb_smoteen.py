from imblearn.combine import SMOTEENN
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score, make_scorer
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from CountFeatureGenerator import *
from TfidfFeatureGenerator import *
from SvdFeatureGenerator import *
from Word2VecFeatureGenerator import *
from SentimentFeatureGenerator import *
from score import *
from generateFeatures import *


def train():
    data_x, data_y, _, _ = build_data()
    test_x, body_ids_test, true_y = build_test_data()

    # Apply SMOTEENN to balance and denoise
    print("🔄 Applying SMOTEENN to balance and clean dataset...")
    smote_enn = SMOTEENN(random_state=2025)
    data_x, data_y = smote_enn.fit_resample(data_x, data_y)

    # Compute class-balanced weights
    sample_weights = compute_sample_weight(class_weight='balanced', y=data_y)

    # Apply PCA
    print("🔻 Applying PCA for dimensionality reduction...")
    pca = PCA(n_components=100, random_state=2025)
    data_x = pca.fit_transform(data_x)
    test_x = pca.transform(test_x)

    # Hyperparameter tuning grid
    param_dist = {
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.02, 0.04],
        'n_estimators': [50, 150, 300],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_alpha': [0, 0.5, 1.0],
        'reg_lambda': [0.5, 1.5, 3.0]
    }

    # Define XGBoost model
    model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=2025,
        n_jobs=-1,
        tree_method='hist'
    )

    # Randomized hyperparameter tuning
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=10,
        scoring=make_scorer(f1_score),
        cv=5,
        verbose=1,
        random_state=2025
    )

    # Train
    search.fit(data_x, data_y, sample_weight=sample_weights)

    print("✅ Best Parameters:", search.best_params_)
    best_model = search.best_estimator_
    best_model.save_model("xgb_model.json")

    # Predict probabilities
    pred_prob_y = best_model.predict_proba(test_x)[:, 1]

    # Tune threshold
    print("🎯 Tuning decision threshold...")
    best_thresh = 0.5
    best_f1 = 0
    for t in np.linspace(0.3, 0.7, 21):
        temp_pred = (pred_prob_y >= t).astype(int)
        temp_f1 = f1_score(true_y, temp_pred)
        if temp_f1 > best_f1:
            best_f1 = temp_f1
            best_thresh = t

    print(f"Best threshold: {best_thresh:.2f} with F1: {best_f1:.4f}")
    pred_y = (pred_prob_y >= best_thresh).astype(int)

    # Evaluate
    report_score(true_y, pred_y)

    # Save predictions
    pd.DataFrame({
        'id': body_ids_test,
        'pred': pred_y
    }).to_csv("xgb_test_predictions.csv", index=False)

    return best_model




params_xgb = {
    'objective': 'binary:logistic',  # Binary classification
    'eval_metric': 'logloss',        # Evaluation metric
    'eta': 0.1,                      # Learning rate
    'max_depth': 5,                  # Maximum depth of a tree
    'min_child_weight': 1,           # Minimum sum of instance weight needed in a child
    'subsample': 0.8,                # Subsample ratio of the training instances
    'colsample_bytree': 0.8,         # Subsample ratio of columns when constructing each tree
    'lambda': 1.0,                   # L2 regularization term on weights
    'alpha': 0.0,                    # L1 regularization term on weights
    'scale_pos_weight': 1,           # Balancing of positive and negative weights
    'seed': 2025,                    # Random seed
    'verbosity': 1                   # Verbosity of printing messages
}


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
    data_x, data_y, _, _ = build_data()
    test_x, body_ids_test, true_y = build_test_data()

    # Apply SMOTEENN to balance and denoise
    print("🔄 Applying SMOTEENN to balance and clean dataset...")
    smote_enn = SMOTEENN(random_state=2025)
    data_x, data_y = smote_enn.fit_resample(data_x, data_y)

    # Compute class-balanced weights
    sample_weights = compute_sample_weight(class_weight='balanced', y=data_y)

    # Apply PCA
    print("🔻 Applying PCA for dimensionality reduction...")
    pca = PCA(n_components=100, random_state=2025)
    data_x = pca.fit_transform(data_x)
    test_x = pca.transform(test_x)

    # Hyperparameter tuning grid
    param_dist = {
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.02, 0.04],
        'n_estimators': [50, 150, 300],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_alpha': [0, 0.5, 1.0],
        'reg_lambda': [0.5, 1.5, 3.0]
    }

    # Define XGBoost model
    model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=2025,
        n_jobs=-1,
        tree_method='hist'
    )

    # Randomized hyperparameter tuning
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=10,
        scoring=make_scorer(f1_score),
        cv=5,
        verbose=1,
        random_state=2025
    )

    # Train
    search.fit(data_x, data_y, sample_weight=sample_weights)

    print("✅ Best Parameters:", search.best_params_)
    best_model = search.best_estimator_
    best_model.save_model("xgb_model_pca&smoteen.json")

    # Predict probabilities
    pred_prob_y = best_model.predict_proba(test_x)[:, 1]

    # Tune threshold
    print("🎯 Tuning decision threshold...")
    best_thresh = 0.5
    best_f1 = 0
    for t in np.linspace(0.3, 0.7, 21):
        temp_pred = (pred_prob_y >= t).astype(int)
        temp_f1 = f1_score(true_y, temp_pred)
        if temp_f1 > best_f1:
            best_f1 = temp_f1
            best_thresh = t

    print(f"Best threshold: {best_thresh:.2f} with F1: {best_f1:.4f}")
    pred_y = (pred_prob_y >= best_thresh).astype(int)

    # Evaluate
    report_score(true_y, pred_y)

    # Save predictions
    pd.DataFrame({
        'id': body_ids_test,
        'pred': pred_y
    }).to_csv("xgb_test_predictions_smoteen.csv", index=False)

    return best_model