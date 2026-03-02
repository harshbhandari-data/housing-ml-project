import pandas as pd
import numpy as np 
from sklearn.model_selection import StratifiedShuffleSplit,cross_val_score
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error


# import the csv
data=pd.read_csv("housing.csv")

#creating test and train set using sklearn
data["income_category"]=pd.cut(data["median_income"],bins=[0,1.5,3.0,4.5,6,np.inf],labels=[1,2,3,4,5])

split=StratifiedShuffleSplit(n_splits=1,test_size=0.2, random_state=42)

for train_index,test_index in split.split(data,data["income_category"]):
    Strat_train_set=data.loc[train_index].drop("income_category",axis=1)
    Strat_test_set=data.loc[test_index].drop("income_category",axis=1)


#now we work on train_set
housing=Strat_train_set.copy()

#now seprate features and labels
housing_labels=housing["median_house_value"].copy()
housing.drop("median_house_value",axis=1,inplace=True)

#now seprate numeric and categorical value
num_attribs=housing.drop("ocean_proximity",axis=1).columns.tolist()
cat_attribs=["ocean_proximity"]

#pipelining  
# numerical pipeline
num_pipeline=Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler())
    ])

#categorical pipeline
cat_pipeline=Pipeline([
    ("encoder",OneHotEncoder(handle_unknown="ignore"))
    ])

#Full pipleline
housing_pipeline=ColumnTransformer([
    ("num",num_pipeline,num_attribs),
    ("cat",cat_pipeline,cat_attribs)
])

#transform data 
housing_prepared=housing_pipeline.fit_transform(housing)
 
'''
 #choosing a model 
 #sometimest the model overfit and may give very less rmse that can be the case of data overfit (it has ovefitted in our given data and may not perform in real world scenario or when new data is provided )leading to poor generalization

 #to ensure that does not happen we will use cross validation from sklearn.model_selection which Instead of training the model once and evaluating on a holdout set, k-fold cross-validation splits the training data into k folds (typically 10), trains the model on k-1 folds, and validates it on the remaining fold. This process repeats k times.

 #a Decision tree 
dec_reg=DecisionTreeRegressor()
dec_reg.fit(housing_prepared,housing_labels)
dec_pred=dec_reg.predict(housing_prepared)

dec_rmse= root_mean_squared_error(housing_labels,dec_pred)
print(f"the rmse of decision tree model is {dec_rmse}")

#evaluating error using cross validation
tree_rmses=-cross_val_score(dec_reg,housing_prepared,housing_labels,scoring="neg_root_mean_squared_error",cv=10)
 #print("decision tree RMSEs : ",tree_rmses)  #print the rmses of 10 folds 
print("cross val prerformance of decion tree is",pd.Series(tree_rmses).mean() )


# b liner regressor
lin_reg=LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)
lin_pred=lin_reg.predict(housing_prepared)
lin_rmses=-cross_val_score(lin_reg,housing_prepared,housing_labels,scoring="neg_root_mean_squared_error",cv=10)
print("cross val performance of linear regressor is ",pd.Series(lin_rmses).mean())
#lin_rmse=root_mean_squared_error(housing_labels,lin_pred)
#print(f"the rmse of linear regressor is {lin_rmse}")
'''


#c  RandomForestRegressor
Random_forest_reg=RandomForestRegressor()
Random_forest_reg.fit(housing_prepared,housing_labels)
Random_forest_pred=Random_forest_reg.predict(housing_prepared)
#Random_forest_rmse=root_mean_squared_error(housing_labels,Random_forest_pred)
#print(f"the rmse of random forest regressor is{Random_forest_rmse}")
Random_forest_rmses=-cross_val_score(Random_forest_reg,housing_prepared,housing_labels,scoring="neg_root_mean_squared_error",cv=10)
print("cross val score of random forest regressor is ",pd.Series(Random_forest_rmses).mean())

# as we saw that random forest regressor has the lowest value hence it is more efficient for this problem so we will use random forest regressor 