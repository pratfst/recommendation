import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import joblib as jb




app = Flask(__name__)


model = jb.load('lightGBM_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    url=request.form.get("URL")
    countrycode=request.form.get("Countrycode")
    

    #prediction=model.predict(pd.DataFrame({'netloc':'amazon.de', 'countryCode':1}, index=[0]).astype({'netloc':'category', 'countryCode':np.int64}))
    
    netloc = url
    countryCode = countrycode
    prediction=model.predict(pd.DataFrame({'netloc':netloc, 'countryCode':countryCode}, index=[0]).astype({'netloc':'category', 'countryCode':np.int64}))
    result={
        "proxy":prediction
    }

    #return jsonify(result)
    return render_template('index.html', prediction_text=' Proxy id : {} '.format(prediction))



if __name__ == "__main__":
    app.run()
