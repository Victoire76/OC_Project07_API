# 1. Library imports
import pandas as pd
from fastapi import FastAPI, Body
import uvicorn
from sklearn.preprocessing import StandardScaler
import joblib,os
import pickle
import json
import shap

# load the model from disk

model = joblib.load('final_model.sav')
#model = joblib.load('lgbm_balance.pkl')
data_api = pd.read_csv("data_api.csv")
#data_api.drop('Unnamed: 0', axis = 1, inplace = True)
data_api.set_index('SK_ID_CURR', inplace = True)

df_sample= pd.read_csv("data_info.csv")
#df_sample.drop('Unnamed: 0', axis = 1, inplace = Trdata_info.csv"ue)
df_sample.set_index('SK_ID_CURR', inplace = True)

def check_id_pred(id_client):
    customers_id_list = list(data_api.index.sort_values())
    if id_client in customers_id_list :
        return True
    else :
        return False
    
def check_id_info(id_client):
    customers_id_list_info = list(df_sample.index.sort_values())
    if id_client in customers_id_list_info :
        return True
    else :
        return False

app = FastAPI()

@app.get('/')
def index():
    return 'Hello, you are accessing an API'

@app.get('/clinet_info')
def get_inof(id_client : int):
    check = check_id_info(id_client)
    if check:
        data_client = df_sample[df_sample.index== id_client]       
        res = data_client.to_dict(orient='records')
        return(res)
    else:
        res = {"This ID doesn't exist"}
        return(res) 

@app.get('/prediction')
def get_prediction(id_client : int):
    check = check_id_pred(id_client)
    if check:
        data_client = data_api[data_api.index== id_client]
        y_pred = model.predict(data_client)
        y_proba = model.predict_proba(data_client)
        y_proba_list = y_proba.tolist()
        res = {
         "prediction": str(y_pred[0]),
         "proba_yes": round(y_proba[0][1],3),
         
          }
        return(res)
    else:
        res = {"This ID doesn't exist"}
        return(res) 

@app.get('/shap/')
def get_shap(id_client : int):
    check = check_id_pred(id_client)
    if check:
        data_client = data_api[data_api.index== id_client]
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(data_client.values)
        expec_value = str(explainer.expected_value[0])
        df_shap = pd.DataFrame({
                 'SHAP value': shap_vals[1][0],
                  'feature': data_client.columns
         })
        df = df_shap.sort_values(by = 'SHAP value', ascending = False).head(10)
        json= df.to_json(orient = 'records')
        res = {
            'expec_value': expec_value,
            'shap_values' : json         
        }        
        return(res)
    else:
        res = {"This ID doesn't exist"}
        return(res)