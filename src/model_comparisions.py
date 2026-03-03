import pandas as pd
import numpy as np 
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# import the csv
data = pd.read_csv("data/housing.csv")

# creating test and train set using stratified split
data["income_category"] = pd.cut(
    data["median_income"],
    bins=[0, 1.5, 3.0, 4.5, 6, np.inf],
    labels=[1, 2, 3, 4, 5]
)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(data, data["income_category"]):
    strat_train_set = data.loc[train_index].drop("income_category", axis=1)
    strat_test_set = data.loc[test_index].drop("income_category", axis=1)

# now we work on train_set
housing = strat_train_set.copy()

# separate features and labels
housing_labels = housing["median_house_value"].copy()
housing.drop("median_house_value", axis=1, inplace=True)

# separate numeric and categorical values
num_attribs = housing.drop("ocean_proximity", axis=1).columns.tolist()
cat_attribs = ["ocean_proximity"]

# numerical pipeline
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# categorical pipeline
cat_pipeline = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# full pipeline
housing_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs)
])

# transform data
housing_prepared = housing_pipeline.fit_transform(housing)

# ===============================
# Model Comparison using Cross Validation
# ===============================

models = {
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42)
}

for name, model in models.items():
    scores = -cross_val_score(
        model,
        housing_prepared,
        housing_labels,
        scoring="neg_root_mean_squared_error",
        cv=10
    )
    print(f"{name} Cross Validation RMSE: {scores.mean()}")