from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score
from score import report_score  # make sure this is implemented
import pandas as pd
import pickle
import joblib 
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
    data_x, data_y, body_ids, target_stance = build_data()
    test_x, body_ids_test, true_y = build_test_data()

    sample_weights = compute_sample_weight(class_weight='balanced', y=data_y)

    # Define hyperparameter search space
    param_dist = {
        'n_estimators': [100,300],
        'max_depth': [10,30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True]
    }

    # Initialize the model
    model = RandomForestClassifier(random_state=2025)

    # Hyperparameter search
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=3,
        scoring='f1',
        cv=3,
        verbose=1,
        random_state=2025,
        n_jobs=-1
    )

    # Train the model
    search.fit(data_x, data_y, sample_weight=sample_weights)

    print("\n Best Parameters:", search.best_params_)
    best_model = search.best_estimator_
    joblib.dump(best_model, "random_forest_model.pkl")

    # Predict on test set
    pred_y = best_model.predict(test_x)

    # Report performance
    report_score(true_y, pred_y)

    # Save predictions
    df_output = pd.DataFrame({
        'id': body_ids_test,
        'pred': pred_y
    })
    df_output.to_csv("random_forest_predictions.csv", index=False)

    return best_model


if __name__ == '__main__':
    train()


# Best Parameters: {'n_estimators': 300, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_depth': 10, 'bootstrap': True}
# -------------------------------------------------
# |               | not_disaster  |   disaster    |
# -------------------------------------------------
# | not_disaster  |      573      |      289      |
# -------------------------------------------------
# |   disaster    |      371      |      277      |
# -------------------------------------------------

# Classification Report:
#                precision    recall  f1-score   support

# not_disaster       0.61      0.66      0.63       862
#     disaster       0.49      0.43      0.46       648

#     accuracy                           0.56      1510
#    macro avg       0.55      0.55      0.55      1510
# weighted avg       0.56      0.56      0.56      1510


#  Binary F1 Score (target: 1): 0.4563