import pandas as pd
import numpy as np 
import os 
import joblib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error

MODEL_FILE="model/model.pkl"
PIPELINE_FILE="model/pipeline.pkl"

def build_pipeline(num_attribs,cat_attribs):
    num_pipeline=Pipeline([
        ("imputer",SimpleImputer(strategy="median")),
        ("scaler",StandardScaler())
    ])
    cat_pipeline=Pipeline([
        ("encoder",OneHotEncoder(handle_unknown="ignore"))
    ])
    full_pipeline=ColumnTransformer([
        ("num",num_pipeline,num_attribs),
        ("cat",cat_pipeline,cat_attribs)
    ])
    return full_pipeline


# Training the model
housing=pd.read_csv("data/housing.csv")

housing['income_category']=pd.cut(
    housing['median_income'],
    bins=[0,1.5,3.0,4.5,6.0,np.inf],
    labels=[1,2,3,4,5]
)

split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

for train_index,test_index in split.split(housing,housing['income_category']):
    strat_test_set=housing.loc[test_index].drop("income_category",axis=1)
    housing=housing.loc[train_index].drop("income_category",axis=1)

housing_labels=housing['median_house_value'].copy()
housing_features=housing.drop('median_house_value',axis=1)

num_attribs=housing_features.drop("ocean_proximity",axis=1).columns.tolist()
cat_attribs=['ocean_proximity']

pipeline=build_pipeline(num_attribs,cat_attribs)

housing_prepared=pipeline.fit_transform(housing_features)

model=RandomForestRegressor(random_state=42)

model.fit(housing_prepared,housing_labels)

# evaluate on test set
test_labels=strat_test_set['median_house_value'].copy()
test_features=strat_test_set.drop("median_house_value",axis=1)

test_prepared=pipeline.transform(test_features)
test_predictions=model.predict(test_prepared)

rmse=root_mean_squared_error(test_labels,test_predictions)
print("the root mean square error of prediction is ",rmse)

joblib.dump(model,MODEL_FILE)
joblib.dump(pipeline,PIPELINE_FILE)

print("YOUR MODEL IS TRAINED")