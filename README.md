# NLP_Disaster
A machine learning pipeline for classifying real disaster tweets from non-disaster tweets — built for the Kaggle competition: [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started).

## Project Overview
This project tackles a binary classification task using Twitter data. The goal is to predict whether a given tweet is about a real disaster (label = 1) or not (label = 0).

We built an end-to-end pipeline that covers:
- Data preprocessing
- Feature generation using TF-IDF, SVD, Word2Vec, and Sentiment analysis
- Training with XGBoost and SVM classifiers
- Evaluation using F1-score, confusion matrix, and classification reports
- Hyperparameter tuning with `RandomizedSearchCV`

---

## 📁 Project Structure
```
NLP_Disaster/
│
├── data/                   # Contains raw and processed CSV files (e.g. train.csv
  ├── train.csv               # Kaggle dataset file (manually added to /data/)
  ├── test.csv                # Kaggle dataset file (optional, for final predictions)

├── helpers.py              # Utility functions like cosine similarity
├── score.py                # Contains scoring utilities like report_score for future evaluation

├── FeatureGenerator.py     # Base class for feature generators
├── CountFeatureGenerator.py
├── TfidfFeatureGenerator.py
├── SvdFeatureGenerator.py
├── Word2VecFeatureGenerator.py
├── SentimentFeatureGenerator.py

├── generateFeatures.py     # Script to generate features and save as .pkl

├── xgb_train_cv.py         # Training script with XGBoost + CV + tuning + GPU support
├── svm_train.py            # (Optional) Training script using SVM

├── xgb_model.json          # Saved trained XGBoost model
├── xgb_test_predictions.csv  # Output predictions
│
└── README.md               # This file
```

---

## How to Use

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

You may need:
- `xgboost>=2.0.0`
- `scikit-learn`
- `pandas`
- `numpy`
- `cupy` (for GPU support)

### 2. Prepare Data
Download `train.csv` from Kaggle and place it under `./data/`.

### 3. Generate Features
```bash
python generateFeatures.py
```

### 4. Train the Model
```bash
python xgb_train_cv.py
```
This will:
- Load generated `.pkl` features
- Tune hyperparameters
- Train XGBoost model
- Output prediction CSV and metrics

### 5. Make Predictions on New Data
Use the saved model `xgb_model.json` and re-run prediction logic with the test set if needed.

---

## Evaluation
We use `report_score()` for evaluation, which prints:
- Confusion Matrix
- Classification Report
- Binary F1 Score

---

## Models
- `XGBoost` (primary)
- `SVM` (experimental, in `svm_train.py`)

---

## Future Improvements
- Expand test set prediction and submission generation
- Add ensemble methods (e.g., stacking)
- Explore transformer-based models (BERT, RoBERTa)

---

## License
This project is under the MIT License.
