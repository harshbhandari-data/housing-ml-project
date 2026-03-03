# California Housing Price Prediction

## 📌 Project Overview
This project predicts median house values using the California Housing dataset.

The goal was to build a clean machine learning workflow including:
- Data preprocessing
- Stratified train-test split
- Model comparison
- Cross-validation
- Model persistence
- Separate training and inference scripts

---

## ⚙️ Project Structure

housing-project/
│
├── data/
│ ├── housing.csv
│ └── input_sample.csv
│
├── model/
│ ├── model.pkl
│ └── pipeline.pkl
│
├── src/
│ ├── train.py
│ ├── predict.py
│ └── model_comparison.py
│
├── requirements.txt
└── README.md

## 🧠 Approach

1. Used **StratifiedShuffleSplit** based on median income.
2. Built preprocessing pipeline using:
   - `SimpleImputer`
   - `StandardScaler`
   - `OneHotEncoder`
3. Compared models:
   - Decision Tree
   - Linear Regression
   - Random Forest
4. Selected RandomForestRegressor based on cross-validation RMSE.
5. Separated training and prediction scripts.
6. Saved trained model using `joblib`.

---
## 📊 Model Performance

- Cross-validation RMSE ≈ 47k
- RandomForest performed better than Linear and Decision Tree.
- Observed model performance plateau during experimentation.

---

## ▶️ How To Run

### Train the model

python src/train.py


### Run prediction

python src/predict.py


Output will be saved as:

data/output.csv




## 📚 What I Learned

During this project, I explored the full ML workflow rather than only training one model.

### 📊 Data Exploration & Visualization
- Analyzed feature distributions and correlations.
- Observed that `median_income` had strong correlation with house prices.
- Understood how skewed data and outliers can influence model performance.

### 🤖 Model Experimentation
- Trained and evaluated:
  - Decision Tree Regressor
  - Linear Regression
  - Random Forest Regressor
- Observed that Decision Tree severely overfitted (very low train RMSE but high cross-validation RMSE).
- Linear Regression underperformed due to limited ability to capture non-linear relationships.
- Random Forest provided better generalization.

### 🔁 Cross-Validation vs Single Split
- Learned why cross-validation gives a more reliable estimate than evaluating only on training data.
- Observed performance plateau even after trying multiple improvements.

### ⚙️ Hyperparameter Tuning
- Experimented with GridSearchCV.
- Learned that hyperparameter tuning does not always significantly reduce error.
- Understood the importance of data signal over excessive tuning.

### 🧪 Feature Engineering
- Tried adding engineered features.
- Learned that additional features do not always improve performance.
- Understood the need to validate improvements using cross-validation.

### 📏 Understanding RMSE
- Compared RMSE across models.
- Learned how overfitting appears through large train vs validation error differences.
- Understood why RMSE is sensitive to large prediction errors.

### 🏗 Project Structuring
- Separated training and inference logic.
- Used pipelines to ensure consistent preprocessing.
- Learned how to structure ML projects cleanly for reproducibility.



## 🔬 Experimentation Journey

The development process involved multiple iterations:

1. Started with a Decision Tree model.
2. Observed strong overfitting.
3. Switched to Linear Regression for comparison.
4. Implemented Random Forest and evaluated using cross-validation.
5. Experimented with feature engineering.
6. Tried hyperparameter tuning using GridSearchCV.
7. Concluded that Random Forest baseline provided stable performance (~47k RMSE).

This iterative experimentation helped build intuition about model behavior and generalization.


## 🚀 Future Improvements

- Experiment with advanced boosting algorithms (XGBoost / LightGBM)
- Try log transformation of target variable
- Perform deeper feature engineering
- Deploy model using Flask or FastAPI

---

### 📎 Summary

This project emphasizes building a structured and reproducible ML workflow while understanding model behavior, evaluation metrics, and generalization rather than only optimizing for lower error.

## 🔧 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib