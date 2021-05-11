
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import preprocessing
from preprocessing import pipeline_preprocessing2

import json
import requests

app = Flask(__name__)


load_model_rfc_IE = pickle.load(open('rfc_IE.pkl', 'rb'))
load_model_rfc_JP = pickle.load(open('rfc_JP.pkl', 'rb'))
load_model_rfc_NS = pickle.load(open('rfc_NS.pkl', 'rb'))
load_model_rfc_TF = pickle.load(open('rfc_TF.pkl', 'rb'))


@app.route('/')
def hello_world():
    return render_template("index.html")



@app.route('/predict',methods=['POST','GET'])
def output():
    try:
        content=request.form['posts']
    
        str=[content]

        test_input = pipeline_preprocessing2.fit_transform(str)
        IE=load_model_rfc_IE.predict(test_input)[0]
        JP=load_model_rfc_JP.predict(test_input)[0]
        NS=load_model_rfc_NS.predict(test_input)[0]
        TF=load_model_rfc_TF.predict(test_input)[0]

        res= IE+" "+NS+" "+TF+" "+" "+JP

        pers=[IE,NS,TF,JP]

        if IE=='Introversion':
            IE='I'
        else:
            IE='E'

        if NS=='Intuition':
            NS='N'
        else:
            NS='S'

        if TF=='Thinking':
            TF='T'
        else:
            TF='F'

        if JP=='Judging':
            JP='J'
        else:
            JP='P'


        res2= IE+NS+TF+JP

        ref_df = pd.read_csv("reference_data/MBTI.csv")
        traits=ref_df[ref_df["type"]==res2]["traits"].values[0]
        careers=ref_df[ref_df["type"]==res2]["career"].values[0]
        em_person=ref_df[ref_df["type"]==res2]["eminent personalities"].values[0]
        name=ref_df[ref_df["type"]==res2]["name"].values[0]


        return render_template('dashboard.html',pred='Your Personality Type is:  {}  '.format(res2) + '{}'.format(name), 
            traits=traits, careers=careers, personalities=em_person, pers=pers)
        #return render_template('index.html', pred=tmp)

    except Exception:
        render_template('error.html')



if __name__=='__main__':
    app.run(debug=True)