import pickle
from flask import Flask,request,render_template,jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

application=Flask(__name__)
app=application

#import ridge regressor and StandardScaler
Logistic_Regression=pickle.load(open('models/lr.pickle','rb'))
standard_scaler=pickle.load(open('models/sc.pkl','rb'))

@app.route("/",methods=['GET','POST'])
def predict_datapoint():
    if request.method=="POST":
        Nitrogen = float(request.form.get('Nitrogen'))
        Phosphorous = float(request.form.get('Phosphorous'))
        Potassium = float(request.form.get('Potassium'))
        Temperature = float(request.form.get('Temperature'))
        Humidity = float(request.form.get('Humidity'))
        PH = float(request.form.get('PH'))
        Rainfall = float(request.form.get('Rainfall'))

        new_data_scaled=standard_scaler.transform([[Nitrogen,Phosphorous,Potassium,Temperature,Humidity,PH,Rainfall]])
        result=Logistic_Regression.predict(new_data_scaled)

        return render_template('index.html',results=result[0])
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)