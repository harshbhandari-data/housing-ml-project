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

if not os.path.exists(MODEL_FILE):
    # Training the model
    housing=pd.read_csv("data/housing.csv")
    housing['income_category']=pd.cut(housing['median_income'],bins=[0,1.5,3.0,4.5,6.0,np.inf],labels=[1,2,3,4,5])
    split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    for train_index,test_index in split.split(housing,housing['income_category']):
        test_set=housing.loc[test_index].drop("income_category",axis=1)
        test_set.to_csv("data/input.csv",index=False)
        housing=housing.loc[train_index].drop("income_category",axis=1)
    housing_labels=housing['median_house_value'].copy()
    housing_features=housing.drop('median_house_value',axis=1)

    num_attribs=housing_features.drop("ocean_proximity",axis=1).columns.tolist()
    cat_attribs=['ocean_proximity']
    
    pipeline=build_pipeline(num_attribs,cat_attribs)
    housing_prepared=pipeline.fit_transform(housing_features)

    model=RandomForestRegressor(random_state=42)
    model.fit(housing_prepared,housing_labels)

    joblib.dump(model,MODEL_FILE)
    joblib.dump(pipeline,PIPELINE_FILE)
    print("YOUR MODEL IS TRAINED")

else:
    #for infetence
    model=joblib.load(MODEL_FILE)
    pipeline=joblib.load(PIPELINE_FILE)

    input_data=pd.read_csv("data/input.csv")
    #separte the labels
    
    input_labels=input_data['median_house_value'].copy()
    input_features=input_data.drop("median_house_value",axis=1)

    #test on only features
    transformed_input=pipeline.transform(input_features)
    predictions=model.predict(transformed_input)
    
    '''#rmse
    #rmse=root_mean_squared_error(input_labels,predictions)
    #print("the root mean square error of prediction is ", rmse)
    #we got rmse of 47197    average acceptable 
'''
    input_data["median_house_value"]=predictions
    input_data.to_csv("data/output.csv",index=False)
    print("Inference is complete . Ouput saved as ouput.csv !!  ")
