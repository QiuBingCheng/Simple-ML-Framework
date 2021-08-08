# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 18:02:31 2021

@author: Jerry
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,BayesianRidge
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.metrics import explained_variance_score,mean_absolute_error,mean_squared_error,r2_score
from data_visualization import plot_true_and_pred
import time
import os
#%%
regression = True 
true_pred_plot = True  #是否要畫true pred plot
grid_search = True     #是否要用grid search找到最佳參數
folder_name = "Boston" 
target = "MEDV"        #目標名稱
data_file = ""  #預設資料夾的第一個file
echo = True     #是否print訊息
#%%
data_folder = f"../dataset/{folder_name}"
if not data_file:
    files = os.listdir(data_folder)
    data_file = [file for file in files if file.split(".")[-1] =="csv"][0]

evalation_folder = f"../evaluation/{folder_name}"
result_path = f"{evalation_folder}/result.csv"
if not os.path.isdir(evalation_folder):
    os.mkdir(evalation_folder)
    if echo:
        print(f"Create folder {evalation_folder}")
        
#%%
#read & split dataset
df = pd.read_csv(f"{data_folder}/{data_file}")
if echo:
    print(f"Read dataset {data_folder}/{data_file}")
    print(f"Dataset shape {df.shape}")
        
y = df[target]
X = df.drop(columns=target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#%%
# initial parameters
svr_paras = {"kernel":'rbf', "C":100,"gamma":0.1,"epsilon":.1}
gb_paras = {"n_estimators":100}

# claim model
regression_models = [LinearRegression(),
                    BayesianRidge(),
                    SVR(**svr_paras),
                    RandomForestRegressor(),
                    GradientBoostingRegressor(**gb_paras),
                    ]

evaluation_metrics = [explained_variance_score,
                      r2_score,
                      mean_absolute_error,
                      mean_squared_error]
                      
columns = ["model","parameters"]
for evluate in evaluation_metrics:
    columns.append(evluate.__name__)
    
results = []

for model in regression_models:
    result = []
    
    model_name = str(model)[:str(model).index("(")]
    #convert parameters to readble string
    para_str = "\n".join([f"{key}={svr_paras[key]}" for key in svr_paras.keys()])
    
    #fit & pred
    if echo:
        print(f"Fitting {model_name}")
        
    start = time.time()
    model.fit(X_train,y_train)
    
    if echo:
        print(f"Fitted {model_name} in {round(time.time()-start,2)}s ")
        
    pred = model.predict(X_test)
    
    #if is_plot
    if true_pred_plot:
        img_path = f"{evalation_folder}/{model_name}.png"
        plot_true_and_pred(y_test,pred,title=model_name,imgpath=img_path)    
        if echo:
            print(f"Save {img_path}")
            
    result = [model_name,para_str]
    for evaluate in evaluation_metrics:
         score = evaluate(y_test,pred)
         result.append(score)
 

    results.append(result)


result_df = pd.DataFrame(results,columns=columns)
result_df.to_csv(result_path,index=False)
if echo:
    print(f"Save {result_path}")


