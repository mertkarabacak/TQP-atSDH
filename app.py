import os
HF_TOKEN = os.getenv("HF_TOKEN")

import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics
from math import sqrt
from scipy import stats as st
from matplotlib import pyplot as plt

from sklearn.linear_model import LogisticRegression

import shap
import gradio as gr
import random
import re
import textwrap
from datasets import load_dataset


#Read data training data.

x1 = load_dataset("mertkarabacak/TQP-atSDH", data_files="mortality_data_train.csv", use_auth_token = HF_TOKEN)
x1 = pd.DataFrame(x1['train'])
x1 = x1.iloc[:, 1:]

x2 = load_dataset("mertkarabacak/TQP-atSDH", data_files="discharge_data_train.csv", use_auth_token = HF_TOKEN)
x2 = pd.DataFrame(x2['train'])
x2 = x2.iloc[:, 1:]

x3 = load_dataset("mertkarabacak/TQP-atSDH", data_files="los_data_train.csv", use_auth_token = HF_TOKEN)
x3 = pd.DataFrame(x3['train'])
x3 = x3.iloc[:, 1:]

x4 = load_dataset("mertkarabacak/TQP-atSDH", data_files="iculos_data_train.csv", use_auth_token = HF_TOKEN)
x4 = pd.DataFrame(x4['train'])
x4 = x4.iloc[:, 1:]

x5 = load_dataset("mertkarabacak/TQP-atSDH", data_files="complications_data_train.csv", use_auth_token = HF_TOKEN)
x5 = pd.DataFrame(x5['train'])
x5 = x5.iloc[:, 1:]


#Read validation data.

x1_valid = load_dataset("mertkarabacak/TQP-atSDH", data_files="mortality_data_valid.csv", use_auth_token = HF_TOKEN)
x1_valid = pd.DataFrame(x1_valid['train'])
x1_valid = x1_valid.iloc[:, 1:]

x2_valid = load_dataset("mertkarabacak/TQP-atSDH", data_files="discharge_data_valid.csv", use_auth_token = HF_TOKEN)
x2_valid = pd.DataFrame(x2_valid['train'])
x2_valid = x2_valid.iloc[:, 1:]

x3_valid = load_dataset("mertkarabacak/TQP-atSDH", data_files="los_data_valid.csv", use_auth_token = HF_TOKEN)
x3_valid = pd.DataFrame(x3_valid['train'])
x3_valid = x3_valid.iloc[:, 1:]

x4_valid = load_dataset("mertkarabacak/TQP-atSDH", data_files="iculos_data_valid.csv", use_auth_token = HF_TOKEN)
x4_valid = pd.DataFrame(x4_valid['train'])
x4_valid = x4_valid.iloc[:, 1:]

x5_valid = load_dataset("mertkarabacak/TQP-atSDH", data_files="complications_data_valid.csv", use_auth_token = HF_TOKEN)
x5_valid = pd.DataFrame(x5_valid['train'])
x5_valid = x5_valid.iloc[:, 1:]

#Define feature names.
f1_names = list(x1.columns)
f1_names = [f1.replace('__', ' - ') for f1 in f1_names]
f1_names = [f1.replace('_', ' ') for f1 in f1_names]

f2_names = list(x2.columns)
f2_names = [f2.replace('__', ' - ') for f2 in f2_names]
f2_names = [f2.replace('_', ' ') for f2 in f2_names]

f3_names = list(x3.columns)
f3_names = [f3.replace('__', ' - ') for f3 in f3_names]
f3_names = [f3.replace('_', ' ') for f3 in f3_names]

f4_names = list(x4.columns)
f4_names = [f4.replace('__', ' - ') for f4 in f4_names]
f4_names = [f4.replace('_', ' ') for f4 in f4_names]

f5_names = list(x5.columns)
f5_names = [f5.replace('__', ' - ') for f5 in f5_names]
f5_names = [f5.replace('_', ' ') for f5 in f5_names]

#Assign unique values as answer options.

unique_TBIPUPILLARYRESPONSE = ['Both reactive', 'One reactive', 'Neither reactive', 'Unknown']
unique_TBIMIDLINESHIFT = ['No', 'Yes', 'Not imaged/unknown']
unique_LOCALIZATION = ['Supratentorial', 'Infratentorial', 'Unknown']
unique_SIZE = ['Tiny (less than 0.6cm thick)', 'Small or moderate (less than 30cc or 0.6-1cm thick)', 'Large, massive, or extensive (more than 30cc, more than 1cm thick', 'Bilateral small or moderate (less than 30cc or 0.6-1cm thick)', 'Bilateral large, massive, or extensive (more than 30cc, more than 1cm thick)']
unique_MECHANISM = ['Fall', 'MVT occupant', 'Struck by or against', 'Other transport or MVT', 'MVT motorcyclist', 'MVT pedestrian', 'Other pedestrian', 'Other pedal cyclist', 'Firearm', 'MVT pedal cyclist', 'Natural or environmental', 'Cut/pierce', 'Machinery', 'Other/unspecified/unknown']
unique_DRGSCR_COCAINE = ['No', 'Yes', 'Not tested']
unique_VERIFICATIONLEVEL = ['Level I Trauma Center', 'Level II Trauma Center', 'Level III Trauma Center', 'Unknown']
unique_BEDSIZE = ['More than 600', '401 to 600', '201 to 400', '200 or fewer']
unique_PRIMARYMETHODPAYMENT = ['Medicare', 'Private/commercial insurance', 'Medicaid', 'Self-pay', 'Not billed', 'Other/Unknown']


#Prepare training data for the outcome 1 (mortality).
y1 = x1.pop('OUTCOME')

#Prepare validation data for the outcome 1 (mortality).
y1_valid = x1_valid.pop('OUTCOME')

#Prepare training data for the outcome 2 (discharge).
y2 = x2.pop('OUTCOME')

#Prepare validation data for the outcome 2 (discharge).
y2_valid = x2_valid.pop('OUTCOME')

#Prepare training data for the outcome 3 (LOS).
y3 = x3.pop('OUTCOME')

#Prepare validation data for the outcome 3 (LOS).
y3_valid = x3_valid.pop('OUTCOME')

#Prepare training data for the outcome 4 (ICU-LOS).
y4 = x4.pop('OUTCOME')

#Prepare validation data for the outcome 4 (ICU-LOS).
y4_valid = x4_valid.pop('OUTCOME')

#Prepare data for the outcome 5 (complications).
y5 = x5.pop('OUTCOME')

#Prepare validation data for the outcome 5 (complications).
y5_valid = x5_valid.pop('OUTCOME')


#Assign hyperparameters.

y2_params =  {'objective': 'binary', 'boosting_type': 'gbdt', 'lambda_l1': 4.834338540250449e-08, 'lambda_l2': 0.06997241053725343, 'num_leaves': 15, 'feature_fraction': 0.5100460294042846, 'bagging_fraction': 0.42648713214413925, 'bagging_freq': 3, 'min_child_samples': 22, 'metric': 'binary_logloss', 'verbosity': -1, 'random_state': 31}
y3_params =  {'objective': 'binary', 'boosting_type': 'gbdt', 'lambda_l1': 0.43069947140430426, 'lambda_l2': 0.010124623249983946, 'num_leaves': 9, 'feature_fraction': 0.4870722075325712, 'bagging_fraction': 0.49967194654478564, 'bagging_freq': 1, 'min_child_samples': 31, 'metric': 'binary_logloss', 'verbosity': -1, 'random_state': 31}
y4_params =  {'objective': 'binary', 'boosting_type': 'gbdt', 'lambda_l1': 0.00520725528228144, 'lambda_l2': 9.85972725587576, 'num_leaves': 8, 'feature_fraction': 0.6203389676645537, 'bagging_fraction': 0.5697730608538535, 'bagging_freq': 4, 'min_child_samples': 22, 'metric': 'binary_logloss', 'verbosity': -1, 'random_state': 31}


#Training models.
from tabpfn import TabPFNClassifier
tabpfn = TabPFNClassifier(device='cpu', N_ensemble_configurations=2)
y1_model = tabpfn

y1_model = y1_model.fit(x1, y1)
y1_explainer = shap.Explainer(y1_model.predict, x1.values)
y1_calib_probs = y1_model.predict_proba(x1_valid)
y1_calib_model = LogisticRegression()
y1_calib_model = y1_calib_model.fit(y1_calib_probs, y1_valid)


from lightgbm import LGBMClassifier
lgb = LGBMClassifier(**y2_params)
y2_model = lgb

y2_model = y2_model.fit(x2, y2)
y2_explainer = shap.Explainer(y2_model.predict, x2.values)
y2_calib_probs = y2_model.predict_proba(x2_valid)
y2_calib_model = LogisticRegression()
y2_calib_model = y2_calib_model.fit(y2_calib_probs, y2_valid)


from lightgbm import LGBMClassifier
lgb = LGBMClassifier(**y3_params)
y3_model = lgb

y3_model = y3_model.fit(x3, y3)
y3_explainer = shap.Explainer(y3_model.predict, x3.values)
y3_calib_probs = y3_model.predict_proba(x3_valid)
y3_calib_model = LogisticRegression()
y3_calib_model = y3_calib_model.fit(y3_calib_probs, y3_valid)


from lightgbm import LGBMClassifier
lgb = LGBMClassifier(**y4_params)
y4_model = lgb

y4_model = y4_model.fit(x4, y4)
y4_explainer = shap.Explainer(y4_model.predict, x4.values)
y4_calib_probs = y4_model.predict_proba(x4_valid)
y4_calib_model = LogisticRegression()
y4_calib_model = y4_calib_model.fit(y4_calib_probs, y4_valid)


from tabpfn import TabPFNClassifier
tabpfn = TabPFNClassifier(device='cpu', N_ensemble_configurations=2)
y5_model = tabpfn

y5_model = y5_model.fit(x5, y5)
y5_explainer = shap.Explainer(y5_model.predict, x5.values)
y5_calib_probs = y5_model.predict_proba(x5_valid)
y5_calib_model = LogisticRegression()
y5_calib_model = y5_calib_model.fit(y5_calib_probs, y5_valid)


output_y1 = (
    """          
        <br/>
        <center>The predicted risk of in-hospital mortality:</center>
        <br/>
        <center><h1>{:.2f}%</h1></center>
"""
)

output_y2 = (
    """          
        <br/>        
        <center>The predicted risk of non-home discharge:</center>
        <br/>        
        <center><h1>{:.2f}%</h1></center>
"""
)

output_y3 = (
    """          
        <br/>        
        <center>The predicted risk of prolonged length of stay:</center>
        <br/>        
        <center><h1>{:.2f}%</h1></center>
"""
)

output_y4 = (
    """          
        <br/>        
        <center>The predicted risk of prolonged length of ICU stay:</center>
        <br/>        
        <center><h1>{:.2f}%</h1></center>
"""
)

output_y5 = (
    """          
        <br/>        
        <center>The predicted risk of major complications:</center>
        <br/>        
        <center><h1>{:.2f}%</h1></center>
"""
)

#Define predict for y1.
def y1_predict(*args):
    df1 = pd.DataFrame([args], columns=x1.columns)
    pos_pred = y1_model.predict_proba(df1)
    pos_pred = y1_calib_model.predict_proba(pos_pred)
    prob = pos_pred[0][1]
    output = output_y1.format(prob * 100)
    return output

#Define predict for y2.
def y2_predict(*args):
    df2 = pd.DataFrame([args], columns=x2.columns)
    pos_pred = y2_model.predict_proba(df2)
    pos_pred = y2_calib_model.predict_proba(pos_pred)        
    prob = pos_pred[0][1]
    output = output_y2.format(prob * 100)
    return output

#Define predict for y3.
def y3_predict(*args):
    df3 = pd.DataFrame([args], columns=x3.columns)
    pos_pred = y3_model.predict_proba(df3)
    pos_pred = y3_calib_model.predict_proba(pos_pred)            
    prob = pos_pred[0][1]
    output = output_y3.format(prob * 100)
    return output

#Define predict for y4.
def y4_predict(*args):
    df4 = pd.DataFrame([args], columns=x4.columns)
    pos_pred = y4_model.predict_proba(df4)
    pos_pred = y4_calib_model.predict_proba(pos_pred)                
    prob = pos_pred[0][1]
    output = output_y4.format(prob * 100)
    return output

#Define predict for y5.
def y5_predict(*args):
    df5 = pd.DataFrame([args], columns=x5.columns)
    pos_pred = y5_model.predict_proba(df5)
    pos_pred = y5_calib_model.predict_proba(pos_pred)                    
    prob = pos_pred[0][1]
    output = output_y5.format(prob * 100)
    return output


#Define function for wrapping feature labels.
def wrap_labels(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_yticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width, break_long_words=break_long_words))
    ax.set_yticklabels(labels, rotation=0)
    

#Define interpret for y1 (mortality).
def y1_interpret(*args):
    df1 = pd.DataFrame([args], columns=x1.columns)
    shap_values1 = y1_explainer(df1).values
    shap_values1 = np.abs(shap_values1)
    shap.bar_plot(shap_values1[0], max_display = 10, show = False, feature_names = f1_names)
    fig = plt.gcf()
    ax = plt.gca()
    wrap_labels(ax, 20)
    ax.figure
    plt.tight_layout()
    fig.set_figheight(7)
    fig.set_figwidth(9)
    plt.xlabel("SHAP value (impact on model output)", fontsize =12, fontweight = 'heavy', labelpad = 8)
    plt.tick_params(axis="y",direction="out", labelsize = 12)
    plt.tick_params(axis="x",direction="out", labelsize = 12)
    return fig

#Define interpret for y2 (discharge).
def y2_interpret(*args):
    df2 = pd.DataFrame([args], columns=x2.columns)
    shap_values2 = y2_explainer(df2).values
    shap_values2 = np.abs(shap_values2)
    shap.bar_plot(shap_values2[0], max_display = 10, show = False, feature_names = f2_names)
    fig = plt.gcf()
    ax = plt.gca()
    wrap_labels(ax, 20)
    ax.figure
    plt.tight_layout()
    fig.set_figheight(7)
    fig.set_figwidth(9)
    plt.xlabel("SHAP value (impact on model output)", fontsize =12, fontweight = 'heavy', labelpad = 8)
    plt.tick_params(axis="y",direction="out", labelsize = 12)
    plt.tick_params(axis="x",direction="out", labelsize = 12)
    return fig

#Define interpret for y3 (LOS).
def y3_interpret(*args):
    df3 = pd.DataFrame([args], columns=x3.columns)
    shap_values3 = y3_explainer(df3).values
    shap_values3 = np.abs(shap_values3)
    shap.bar_plot(shap_values3[0], max_display = 10, show = False, feature_names = f3_names)
    fig = plt.gcf()
    ax = plt.gca()
    wrap_labels(ax, 20)
    ax.figure
    plt.tight_layout()
    fig.set_figheight(7)
    fig.set_figwidth(9)
    plt.xlabel("SHAP value (impact on model output)", fontsize =12, fontweight = 'heavy', labelpad = 8)
    plt.tick_params(axis="y",direction="out", labelsize = 12)
    plt.tick_params(axis="x",direction="out", labelsize = 12)
    return fig

#Define interpret for y4 (ICU LOS).
def y4_interpret(*args):
    df4 = pd.DataFrame([args], columns=x4.columns)
    shap_values4 = y4_explainer(df4).values
    shap_values4 = np.abs(shap_values4)
    shap.bar_plot(shap_values4[0], max_display = 10, show = False, feature_names = f4_names)
    fig = plt.gcf()
    ax = plt.gca()
    wrap_labels(ax, 20)
    ax.figure
    plt.tight_layout()
    fig.set_figheight(7)
    fig.set_figwidth(9)
    plt.xlabel("SHAP value (impact on model output)", fontsize =12, fontweight = 'heavy', labelpad = 8)
    plt.tick_params(axis="y",direction="out", labelsize = 12)
    plt.tick_params(axis="x",direction="out", labelsize = 12)
    return fig

#Define interpret for y5 (complications).
def y5_interpret(*args):
    df5 = pd.DataFrame([args], columns=x5.columns)
    shap_values5 = y5_explainer(df5).values
    shap_values5 = np.abs(shap_values5)
    shap.bar_plot(shap_values5[0], max_display = 10, show = False, feature_names = f5_names)
    fig = plt.gcf()
    ax = plt.gca()
    wrap_labels(ax, 20)
    ax.figure
    plt.tight_layout()
    fig.set_figheight(7)
    fig.set_figwidth(9)
    plt.xlabel("SHAP value (impact on model output)", fontsize =12, fontweight = 'heavy', labelpad = 8)
    plt.tick_params(axis="y",direction="out", labelsize = 12)
    plt.tick_params(axis="x",direction="out", labelsize = 12)
    return fig

with gr.Blocks(title = "TQP-atSDH") as demo:
        
    gr.Markdown(
        """
    <br/>
    <center><h2>NOT FOR CLINICAL USE</h2><center>    
    <br/>    
    <center><h1>Acute Traumatic Subdural Hematoma Outcomes</h1></center>
    <center><h2>Prediction Tool</h2></center>
    <br/>
    <center><h3>This web application should not be used to guide any clinical decisions.</h3><center>
    <br/>
    <center><i>The publication describing the details of this prediction tool will be posted here upon the acceptance of publication.</i><center>
        """
    )

    gr.Markdown(
        """
        <center><h3>Model Performances</h3></center>
          <div style="text-align:center;">
          <table>
          <tr>
            <th>Outcome</th>
            <th>Algorithm</th>
            <th>Weighted Precision</th>
            <th>Weighted Recall</th>
            <th>Weighted AUPRC</th>
            <th>Balanced Accuracy</th>
            <th>AUROC</th>
            <th>Brier Score</th>
          </tr>
          <tr>
            <td>Mortality</td>
            <td>TabPFN</td>
            <td>0.934 (0.929 - 0.939)</td>
            <td>0.827 (0.820 - 0.834)</td>
            <td>0.597 (0.588 - 0.606)</td>
            <td>0.813 (0.806 - 0.820)</td>
            <td>0.934 (0.880 - 0.946)</td>
            <td>0.043 (0.039 - 0.047)</td>             
          </tr>
          <tr>
            <td>Non-home Discharges</td>
            <td>LightGBM</td>
            <td>0.701 (0.691 - 0.711)</td>
            <td>0.683 (0.673 - 0.693)</td>
            <td>0.686 (0.676 - 0.696)</td>
            <td>0.692 (0.682 - 0.702)</td>
            <td>0.772 (0.751 - 0.772)</td>
            <td>0.196 (0.187 - 0.205)</td>             
          </tr>
          <tr>
            <td>Prolonged LOS</td>
            <td>LightGBM</td>
            <td>0.794 (0.786 - 0.802)</td>
            <td>0.693 (0.684 - 0.702)</td>
            <td>0.416 (0.407 - 0.425)</td>
            <td>0.684 (0.675 - 0.693)</td>
            <td>0.762 (0.745 - 0.767)</td>
            <td>0.135 (0.128 - 0.142)</td>             
          </tr>
          <tr>
            <td>Prolonged ICU-LOS</td>
            <td>LightGBM</td>
            <td>0.817 (0.807 - 0.827)</td>
            <td>0.732 (0.721 - 0.743)</td>
            <td>0.486 (0.474 - 0.498)</td>
            <td>0.729 (0.718 - 0.740)</td>
            <td>0.807 (0.789 - 0.814)</td>
            <td>0.126 (0.118 - 0.134)</td>             
          </tr>
          <tr>
            <td>Major Complications</td>
            <td>TabPFN</td>
            <td>0.945 (0.931 - 0.959)</td>
            <td>0.706 (0.678 - 0.734)</td>
            <td>0.106 (0.087 - 0.125)</td>
            <td>0.710 (0.682 - 0.738)</td>
            <td>0.800 (0.695 - 0.834)</td>
            <td>0.039 (0.027 - 0.051)</td>             
          </tr>          
        </table>
        </div>
        """
    )    

    with gr.Row():

        with gr.Column():

            Age = gr.Slider(label="Age", minimum = 18, maximum = 99, step = 1, value = 37)

            Weight = gr.Slider(label = "Weight (in kilograms)", minimum = 20, maximum = 200, step = 1, value = 75)
            
            Height = gr.Slider(label = "Height (in centimeters)", minimum = 100, maximum = 250, step = 1, value = 175)
            
            Systolic_Blood_Pressure = gr.Slider(label = "Systolic Blood Pressure", minimum = 50, maximum = 250, step = 1, value = 135)

            Pulse_Rate = gr.Slider(label = "Pulse Rate", minimum=20, maximum=250, step=1, value = 75)

            Pulse_Oximetry = gr.Slider(label = "Pulse Oximetry", minimum = 50, maximum = 100, step = 1, value = 99)

            Respiratory_Rate = gr.Slider(label = "Respiratory Rate", minimum = 4, maximum = 45, step = 1, value = 18)

            GCS__Verbal = gr.Slider(label = "GCS - Verbal", minimum = 1, maximum = 5, step = 1, value = 5)

            Total_GCS = gr.Slider(label = "GCS - Total", minimum = 1, maximum = 15, step = 1, value = 15)
            
            Pupillary_Response = gr.Radio(label = "Pupillary Response", choices = unique_TBIPUPILLARYRESPONSE, type = 'index', value = 'Both reactive')
            
            Midline_Shift = gr.Radio(label = "Midline Shift", choices = unique_TBIMIDLINESHIFT, type = 'index', value = 'No')
            
            Bleeding_Localization = gr.Radio(label = "Bleeding Localization", choices = unique_LOCALIZATION, type = 'index', value = 'Supratentorial')   
            
            Bleeding_Size = gr.Radio(label = "Bleeding Size", choices = unique_SIZE, type = 'index', value = 'Tiny (less than 0.6cm thick)')
                        
            Days_from_Incident_to_ED_or_Hospital_Arrival = gr.Slider(label = "Days from Incident to ED or Hospital Arrival", minimum = 0, maximum = 31, step = 1, value = 0)

            Mechanism_of_Injury = gr.Dropdown(label = "Mechanism of Injury", choices = unique_MECHANISM, type = 'index', value = 'Fall')
                     
            Drug_Screen__Cocaine = gr.Radio(label = "Drug Screen - Cocaine", choices = unique_DRGSCR_COCAINE, type = 'index', value = 'No')

            Blood_Transfusion = gr.Slider(label = "Blood Transfusion (mL)", minimum = 0, maximum = 5000, step = 50, value = 0)
            
            ACS_Verification_Level = gr.Radio(label = "ACS Verification Level", choices = unique_VERIFICATIONLEVEL, type = 'index', value = 'Level I Trauma Center')
                        
            Facility_Bed_Size = gr.Radio(label = "Facility Bed Size", choices = unique_BEDSIZE, type = 'index', value = 'More than 600')
            
            Primary_Method_of_Payment = gr.Dropdown(label = "Primary Method of Payment", choices = unique_PRIMARYMETHODPAYMENT, type = 'index', value = 'Private/commercial insurance')
            
        with gr.Column():
            
            with gr.Box():
                
                gr.Markdown(
                    """
                    <center> <h2>Mortality</h2> </center>
                    <center> This model uses the TabPFN algorithm with the following features: </center>
		    <center> <i>Age, Weight, Height, Systolic Blood Pressure, Pulse Rate, Pulse Oximetry, Respiratory Rate, GCS - Verbal, Total GCS, Midline Shift, Blood Transfusion</i> </center>
                    <br/>
                    """
                    )
                
                with gr.Row():
                    y1_predict_btn = gr.Button(value="Predict")
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )
                
                label1 = gr.Markdown()
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )
                
                with gr.Row():
                    y1_interpret_btn = gr.Button(value="Explain")
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )
                
                plot1 = gr.Plot()
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )
                
            with gr.Box():
                gr.Markdown(
                    """
                    <center> <h2>Discharge Disposition</h2> </center>
                    <center> This model uses the LightGBM algorithm with the following features: </center>
		    <center> <i>Age, Weight, Systolic Blood Pressure, Pulse Rate, Respiratory Rate, Total GCS, Bleeding Size, Mechanism of Injury, Blood Transfusion, Primary Method of Payment</i> </center>
                    <br/>
                    """
                    )
                
                with gr.Row():
                    y2_predict_btn = gr.Button(value="Predict")
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )
                
                label2 = gr.Markdown()
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )
                
                with gr.Row():
                    y2_interpret_btn = gr.Button(value="Explain")
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )
                
                plot2 = gr.Plot()
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )
                
            with gr.Box():
                
                gr.Markdown(
                    """
                    <center> <h2>Prolonged Length of Stay</h2> </center>
                    <center> This model uses the LightGBM algorithm with the following features: </center>
		    <center> <i>Age, Weight, Height, Systolic Blood Pressure, Pulse Rate, Respiratory Rate, GCS - Verbal, Total GCS, Pupillary Response, Bleeding Size, Days from Incident to ED or Hospital Arrival, Blood Transfusion, Facility Bed Size</i> </center>
                    <br/>
                    """
                    )
                
                with gr.Row():
                    y3_predict_btn = gr.Button(value="Predict")
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )
                
                label3 = gr.Markdown()
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )
                
                with gr.Row():
                    y3_interpret_btn = gr.Button(value="Explain")
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )
                
                plot3 = gr.Plot()
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )  

            with gr.Box():
                gr.Markdown(
                    """
                    <center> <h2>Prolonged Length of ICU Stay</h2> </center>
                    <center> This model uses the LightGBM algorithm with the following features: <center>
		    <center> <i>Age, Weight, Height, Systolic Blood Pressure, Pulse Rate, Respiratory Rate, Total GCS, Midline Shift, Bleeding Localization, Days from Incident to ED or Hospital Arrival, Blood Transfusion, ACS Verification Level</i> </center>
                    <br/>
                    """
                )
                with gr.Row():
                    y4_predict_btn = gr.Button(value="Predict")
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )
                
                label4 = gr.Markdown()
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )
                
                with gr.Row():
                    y4_interpret_btn = gr.Button(value="Explain")
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )
                
                plot4 = gr.Plot()
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )
                
            with gr.Box():
                gr.Markdown(
                    """
                    <center> <h2>Major Complications</h2> </center>
                    <center> This model uses the TabPFN algorithm with the following features: </center> 
		    <center> <i>Age, Weight, Height, Systolic Blood Pressure, Pulse Rate, Respiratory Rate, Total GCS, Pupillary Response, Midline Shift, Bleeding Size, Days from Incident to ED or Hospital Arrival, Blood Transfusion, Drug Screen - Cocaine, ACS Verification Level</i> </center>
                    <br/>
                    """
                    )
                
                with gr.Row():
                    y5_predict_btn = gr.Button(value="Predict")
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )
                
                label5 = gr.Markdown()
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )
                
                with gr.Row():
                    y5_interpret_btn = gr.Button(value="Explain")
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )
                
                plot5 = gr.Plot()
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )                
                                
                y1_predict_btn.click(
                    y1_predict,
                    inputs = [Age, Weight, Height, Systolic_Blood_Pressure, Pulse_Rate, Pulse_Oximetry, Respiratory_Rate, GCS__Verbal, Total_GCS, Midline_Shift, Blood_Transfusion,],
                    outputs = [label1]
                )

                y2_predict_btn.click(
                    y2_predict,
                    inputs = [Age, Weight, Systolic_Blood_Pressure, Pulse_Rate, Respiratory_Rate, Total_GCS, Bleeding_Size, Mechanism_of_Injury, Blood_Transfusion, Primary_Method_of_Payment,],
                    outputs = [label2]
                )
                
                y3_predict_btn.click(
                    y3_predict,
                    inputs = [Age, Weight, Height, Systolic_Blood_Pressure, Pulse_Rate, Respiratory_Rate, GCS__Verbal, Total_GCS, Pupillary_Response, Bleeding_Size, Days_from_Incident_to_ED_or_Hospital_Arrival, Blood_Transfusion, Facility_Bed_Size,],
                    outputs = [label3]
                )

                y4_predict_btn.click(
                    y4_predict,
                    inputs = [Age, Weight, Height, Systolic_Blood_Pressure, Pulse_Rate, Respiratory_Rate, Total_GCS, Midline_Shift, Bleeding_Localization, Days_from_Incident_to_ED_or_Hospital_Arrival, Blood_Transfusion, ACS_Verification_Level,],
                    outputs = [label4]
                )
                
                y5_predict_btn.click(
                    y5_predict,
                    inputs = [Age, Weight, Height, Systolic_Blood_Pressure, Pulse_Rate, Respiratory_Rate, Total_GCS, Pupillary_Response, Midline_Shift, Bleeding_Size, Days_from_Incident_to_ED_or_Hospital_Arrival, Blood_Transfusion, Drug_Screen__Cocaine, ACS_Verification_Level,],
                    outputs = [label5]
                )

                y1_interpret_btn.click(
                    y1_interpret,
                    inputs = [Age, Weight, Height, Systolic_Blood_Pressure, Pulse_Rate, Pulse_Oximetry, Respiratory_Rate, GCS__Verbal, Total_GCS, Midline_Shift, Blood_Transfusion,],
                    outputs = [plot1],
                )
                
                y2_interpret_btn.click(
                    y2_interpret,
                    inputs = [Age, Weight, Systolic_Blood_Pressure, Pulse_Rate, Respiratory_Rate, Total_GCS, Bleeding_Size, Mechanism_of_Injury, Blood_Transfusion, Primary_Method_of_Payment,],
                    outputs = [plot2],
                )

                y3_interpret_btn.click(
                    y3_interpret,
                    inputs = [Age, Weight, Height, Systolic_Blood_Pressure, Pulse_Rate, Respiratory_Rate, GCS__Verbal, Total_GCS, Pupillary_Response, Bleeding_Size, Days_from_Incident_to_ED_or_Hospital_Arrival, Blood_Transfusion, Facility_Bed_Size,],
                  outputs = [plot3],
                )
                
                y4_interpret_btn.click(
                    y4_interpret,
                    inputs = [Age, Weight, Height, Systolic_Blood_Pressure, Pulse_Rate, Respiratory_Rate, Total_GCS, Midline_Shift, Bleeding_Localization, Days_from_Incident_to_ED_or_Hospital_Arrival, Blood_Transfusion, ACS_Verification_Level,],
                    outputs = [plot4],
                )
                
                y5_interpret_btn.click(
                    y5_interpret,
                    inputs = [Age, Weight, Height, Systolic_Blood_Pressure, Pulse_Rate, Respiratory_Rate, Total_GCS, Pupillary_Response, Midline_Shift, Bleeding_Size, Days_from_Incident_to_ED_or_Hospital_Arrival, Blood_Transfusion, Drug_Screen__Cocaine, ACS_Verification_Level,],
                    outputs = [plot5],
                )

    gr.Markdown(
                """    
                <center><h2>Disclaimer</h2>
                <center> 
                The American College of Surgeons National Trauma Data Bank (ACS-NTDB) and the hospitals participating in the ACS-NTDB are the source of the data used herein; they have not been verified and are not responsible for the statistical validity of the data analysis or the conclusions derived by the authors. The predictive tool located on this web page is for general health information only. This prediction tool should not be used in place of professional medical service for any disease or concern. Users of the prediction tool shouldn't base their decisions about their own health issues on the information presented here. You should ask any questions to your own doctor or another healthcare professional. The authors of the study mentioned above make no guarantees or representations, either express or implied, as to the completeness, timeliness, comparative or contentious nature, or utility of any information contained in or referred to in this prediction tool. The risk associated with using this prediction tool or the information in this predictive tool is not at all assumed by the authors. The information contained in the prediction tools may be outdated, not complete, or incorrect because health-related information is subject to frequent change and multiple confounders. No express or implied doctor-patient relationship is established by using the prediction tool. The prediction tools on this website are not validated by the authors. Users of the tool are not contacted by the authors, who also do not record any specific information about them. You are hereby advised to seek the advice of a doctor or other qualified healthcare provider before making any decisions, acting, or refraining from acting in response to any healthcare problem or issue you may be experiencing at any time, now or in the future. By using the prediction tool, you acknowledge and agree that neither the authors nor any other party are or will be liable or otherwise responsible for any decisions you make, actions you take, or actions you choose not to take as a result of using any information presented here.
                <br/>
                <h4>By using this tool, you accept all of the above terms.<h4/>
                </center>
                """
    )                
                
demo.launch()