import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
#from sklearn.linear_model import Ridge

application = Flask(__name__) 
app=application


# Load the pre-trained Ridge regression model
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
      Tempereture = float(request.form.get('Temperature'))
      RH = float(request.form.get('RH'))
      WS = float(request.form.get('WS'))
      Rain = float(request.form.get('Rain'))
      FFMC = float(request.form.get('FFMC'))
      DMC = float(request.form.get('DMC'))
      ISI = float(request.form.get('ISI'))
      Classes = float(request.form.get('Classes'))
      Region = float(request.form.get('Region'))

      new_data_scaled = standard_scaler.transform([[Tempereture, RH, WS, Rain, FFMC, DMC, ISI, Classes, Region]])
      result = ridge_model.predict(new_data_scaled)

      return render_template('home.html', results=result[0])

    else:
      return render_template('home.html')

      

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")

# Note: The above code is a simple Flask application setup.
# It includes a route that returns a welcome message.
# Make sure to install the required packages using:
# pip install -r requirements.txt
# Ensure that the Flask application runs on the specified host and port.
# This code is a basic Flask application that serves as a starting point for deploying a Ridge regression model.