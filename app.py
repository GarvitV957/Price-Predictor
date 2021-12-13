from flask import Flask,render_template,request
import pickle
import numpy as np

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    features=[float(x) for x in request.form.values()]
    arr=[np.array(features)]
    pred=model.predict(arr)
    output=round(pred[0],3)
    return render_template('home.html',prediction_text='Predicted price in (1000 $) is {} '.format(output))

if __name__ == "__main__":
    app.run(debug=True)