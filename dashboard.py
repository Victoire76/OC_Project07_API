import streamlit as st
import streamlit.components.v1 as components
from streamlit_echarts import st_echarts
from PIL import Image
import pandas as pd
import numpy as np
import altair as alt
import joblib
from lightgbm import LGBMClassifier as lgb
import shap
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns


# data for client's info
data_info = pd.read_csv("data_info.csv")
#data_info.drop('Unnamed: 0', axis = 1, inplace = True)
data_info.set_index('SK_ID_CURR', inplace= True)
 
# data for model   
data_api = pd.read_csv("data_api.csv")
#data_api.drop('Unnamed: 0', axis = 1, inplace = True)
data_api.set_index('SK_ID_CURR', inplace = True)
#model = joblib.load('final_model.sav')
ml_model = joblib.load('lgbm_balance.pkl')
features = data_api.columns


mylist = data_info.index.tolist()
best_threshold= 0.49

def get_info(id_client): 
    data_client = data_info[data_info.index== id_client]
    return(data_client)

def get_prediction(id_client):
    data_client = data_api[data_api.index== id_client]
    y_pred = ml_model.predict(data_client)
    y_proba = ml_model.predict_proba(data_client)
    y_proba_list = y_proba.tolist()
    res = {
         "prediction":  int(y_pred[0]),
         "proba_yes": round(y_proba[0][1],3)
          }
    prediction = int(y_pred[0])
    probabilty = round(y_proba[0][1],3)
    return(prediction, probabilty)

    #Plot distribution d'une variable vs Target
def plot_distribution_comp(feature,value):
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    fig, ax = plt.subplots()
    #plt.figure(figsize=(20,6))
    t0 = data_info.loc[data_info['TARGET'] == 0]
    t1 = data_info.loc[data_info['TARGET'] == 1]
    ax= sns.kdeplot(t0[feature].dropna(), color='green', bw_adjust=0.5, label="Credit paid")
    ax= sns.kdeplot(t1[feature].dropna(), color='red', bw_adjust=0.5, label="Credit not paid")
    plt.title("Distribution of %s" % feature, fontsize= 20)
    plt.axvline(value, color='b' ,linewidth = 0.8, alpha= 0.8)
    plt.legend(fontsize=10)
    plt.show()

def get_shap_explainer(id_client):    
    #shap_client = data_api[data_api.index==id_client]
    explainer = shap.TreeExplainer(ml_model)
    shap_vals = explainer.shap_values(data_api)
    expected_vals = explainer.expected_value
    return(shap_vals, expected_vals)    

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height) 
    
st.set_page_config(page_title="Project 7 Dashboard",
                   initial_sidebar_state="expanded",
                   layout="wide")
st.markdown("Creator : **_Victoire MOHEBI_**")

# Side bar
with st.sidebar:
    image_HC = Image.open("logo.jpg")
    st.image(image_HC, width=300)

# CHECKBOX 
home = st.sidebar.checkbox("Home Page")
model = st.sidebar.checkbox("Model information")
customer_info= st.sidebar.checkbox("Customers")
customer_result = st.sidebar.checkbox("Prediction result")

if model :   
    menue = ['Model', 'Feature importance', 'Shap',]
    choice = st.selectbox('Menue', menue)
    if choice == 'Model' : 
        st.title('Model Information')
        st.write('ML model : LGBMClassifier')
        st.write('We have used LGBMClassifier as a machine learning model to classify customers.')
        col1, col2= st.columns(2) 
        with col1 :
                auc_style = '<p style="font-family:Arial; color:#0066cc; font-size: 24px;">AUC_ROC%</p>'
                st.markdown(auc_style, unsafe_allow_html=True)
                roc_auc = 78
                col1.metric('', value=roc_auc)
        with col2:                
                fscore_style = '<p style="font-family:Arial; color:#0066cc; font-size: 24px;">F5_Score%</p>'
                st.markdown(fscore_style, unsafe_allow_html=True)
                f5_score = 60
                col2.metric('', value= f5_score)

    elif choice == 'Feature importance':
        st.title('Feature Importance')
        st.write("Top 20 of the most important features for the model")
        fi = Image.open("feature_importance.jpg")
        st.image(fi)

    else :
        st.title('Shaply Values') 
        st.write("Top 20 features of shaply values the model")
        shap_plot = Image.open("shap_values.jpg")
        st.image(shap_plot)

               
elif customer_info:  
    st.subheader("Select a customer ID")
    choice_id = st.selectbox('', mylist) 
    for i in mylist :
        if choice_id == i:
            customer_df = get_info(choice_id)                
            target, probability_default = get_prediction(choice_id)                        
            customer_dict = customer_df.to_dict('index') 
            style_title = '<p style="font-family:Arial; font-weight: bold;color:Black; font-size: 30px;">Customer basic information</p>'
            st.markdown(style_title,unsafe_allow_html=True)
            #text =  "Customer's basic information"
            #st.markdown(f'<p style="color:#0066cc;">{text}</p>', unsafe_allow_html=True)
            st.write(customer_dict)
            if target == 1 :  
                res, score = st.columns(2)
                with res:
                    #st.success("CREDIT REFUSED ! ", icon= '❌')
                    style1 = '<p style="font-family:Arial; color:Red; font-size: 24px;">Credit refsed</p>'
                    st.markdown(style1, unsafe_allow_html=True)                     
                     
                with score : 
                    score_style1= '<p style="font-family:Arial; color:Red; font-size: 24px;">SCORE:</p>'
                    st.markdown(score_style1, unsafe_allow_html=True)  
                    st.metric('',value= probability_default)
                    #st.markdown(f'<p style="color:#FF0000;font-size:24px;">{probability_default}</p>', unsafe_allow_html=True) 
                    #st.write(probability_default)
                    
            else :
                res2, score2 = st.columns(2)   
                with res2: 
                    #res_t2= 'Credit accepted'    
                    style0 = '<p style="font-family:Arial; color:Green; font-size: 24px;">Credit accepted</p>'
                    #st.markdown(f'<p style="color:#008000;font-size:24px;">{res_t2}</p>', unsafe_allow_html=True) 
                    st.markdown(style0,unsafe_allow_html=True)
                    #st.success("CREDIT ACCEPTED ! ", icon="✅")  
                with score2: 
                    score_style0= '<p style="font-family:Arial; color:Green; font-size: 24px;">SCORE:</p>'
                    st.markdown(score_style0,unsafe_allow_html=True)
                    st.metric('',value = probability_default)
                    #st.write(probability_default)         
 
            feauture_select = ['AMT_ANNUITY','AMT_GOODS_PRICE','ANNUITY_INCOME%','PAYMENT_RATE',
                                    'AGE_CLIENT','YEARS_EMPLOYED', 'CNT_FAM_MEMBERS']  
            f_style= '<p style="font-family:Arial; font-weight: bold;color:Black; font-size: 30px;">Select the feature</p>'
            st.markdown(f_style, unsafe_allow_html=True)
            #st.subheader("Which feature do you want to show?")
            feat_to_show = st.selectbox('', options= feauture_select)
            for var in feauture_select :
                if feat_to_show == var :  
                    val = int(customer_df[var].values)               
                    fig = plot_distribution_comp(var,val)  
                    st.set_option('deprecation.showPyplotGlobalUse', False)        
                    st.pyplot(fig)
                    
                
elif customer_result : 
    st.title("Select an ID")
    choice_id = st.selectbox('Choose a customer ID', options = mylist, index=0)  
    for i in mylist:   
        if choice_id == i: 
            client_info = get_info(choice_id)
            target, probability_default = get_prediction(choice_id)
            client_info = get_info(choice_id)       
            if target == 1 :
                res_title1 = '<p style="font-family:Arial; font-weight: bold;color:Red; font-size: 30px;">Credit Refused</p>'
                st.markdown(res_title1,unsafe_allow_html=True)
               # st.markdown(f'<p style="color:#FF0000;font-size:24px;">{res_t}</p>', unsafe_allow_html=True) 
                fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = 100*probability_default,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Probability of default", 'font': {'size': 24}},       
                gauge = {
                'axis': {'range': [None, 100]},
                    'bar': {'color': "red"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray"              
                    }))
                st.plotly_chart(fig, use_container_width=True) 
                shap_values1, expected_values1 = get_shap_explainer(i) 
                shap_title1 = '<p style="font-family:Arial; font-weight: bold;color:Black; font-size: 30px;">Shap Force Plot</p>'
                st.markdown(shap_title1,unsafe_allow_html=True)
            
                st_shap(shap.force_plot(expected_values1[1], shap_values1[0][1],data_api[data_api.index==i]))
                #st.set_option('deprecation.showPyplotGlobalUse', False) 
                #st.plotly_chart(shap_force_plot1,use_container_width=True)
                
                                                        
            else : 
                res_title2 = '<p style="font-family:Arial; font-weight: bold;color:Green; font-size: 30px;">Credit Accepted</p>'
                st.markdown(res_title2,unsafe_allow_html=True)
                fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = 100*probability_default,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Probability of default", 'font': {'size': 24}},       
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "green"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray"              
                    }))
                st.plotly_chart(fig, use_container_width=True)
                shap_values0, expected_values0 = get_shap_explainer(i) 
                shap_title0 = '<p style="font-family:Arial; font-weight: bold;color:Black; font-size: 30px;">Shap Force Plot</p>'
                st.markdown(shap_title0,unsafe_allow_html=True)
                st_shap(shap.force_plot(expected_values0[0], shap_values0[0][0],data_api[data_api.index==i]))
                #st.plotly_chart(shap_force_plot0,use_container_width=True)
                #st.set_option('deprecation.showPyplotGlobalUse', False) 
                                            
else: 
    st.header('Welcome to Home Credit Default Risk Prediction')
    image = Image.open("home credit.jpg")
    col1, col2, col3 = st.columns([1,10,1])    
    with col1:
        st.write("")
    with col2:
            st.image(image, width=600)
    with col3:
        st.write("")
