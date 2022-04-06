from flask import Flask , render_template , request
import pickle
import numpy as np


app = Flask(__name__)

filename = 'diabetes_prediction_pickle'
model = pickle.load(open(filename,'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    Pregnancies = float(request.form['Pregnancies'])
    Glucose  = float(request.form['Glucose'])
    BloodPressure = float(request.form['BloodPressure'])
    SkinThickness = float(request.form['SkinThickness'])
    Insulin = float(request.form['Insulin'])
    BMI = float(request.form['BMI'])
    DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
    Age = float(request.form['Age'])
    
    data = np.array([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction, Age]])
    my_prediction = model.predict(data)

    return render_template('index.html',
    Pregnancies = str(Pregnancies),
    Glucose = str(Glucose),
    BloodPressure = str(BloodPressure),
    SkinThickness = str(SkinThickness),
    Insulin = str(Insulin),
    BMI = str(BMI),
    DiabetesPedigreeFunction = str(DiabetesPedigreeFunction),
    Age = str(Age),
    prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)