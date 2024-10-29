from flask import Flask, render_template as Render_Template, request as Request, jsonify as JSONify, send_from_directory as Send_From_Directory
import joblib as JobLib
import numpy as NP
import warnings as Warnings

# Suppress Specific Warning About Feature Names.
Warnings.filterwarnings("ignore", category=UserWarning)

App = Flask(__name__, template_folder='Templates', static_folder='Static')

# Load The Model And Scaler.
Model = JobLib.load('Models/Best_SVC_Wine_Model.pkl')
Scaler = JobLib.load('Models/Scaler.pkl')

@App.route('/')
def Home():
    return Render_Template('Index.html')

@App.route('/predict', methods=['POST'])
def Predict():
    if Request.method == 'POST':
        try:
            # Extract Form Data And Transform It.
            Features = [float(X) for X in Request.form.values()]
            Features = NP.array(Features).reshape(1, -1)
            Scaled_Features = Scaler.transform(Features)
            Prediction = Model.predict(Scaled_Features)

            # Map Prediction To Quality Level.
            Quality = int(Prediction[0])
            return Render_Template('Index.html', prediction_text=f'Predicted Wine Quality: {Quality}')

        except Exception as E:
            return JSONify({'error': str(E)})

if __name__ == '__main__':
    App.run(debug=True)

