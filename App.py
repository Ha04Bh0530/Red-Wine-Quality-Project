from flask import Flask, render_template as Render_Template, request as Request, jsonify as JSONify, send_from_directory as Send_From_Directory
import joblib as JobLib
import numpy as NP
import warnings as Warnings

# Suppress specific warning about feature names.
Warnings.filterwarnings("ignore", category=UserWarning)

# Initialize Flask application.
App = Flask(__name__, template_folder='Templates', static_folder='Static')

# Load the trained model and scaler.
Model = JobLib.load('Models/Best_SVC_Wine_Model.pkl')
Scaler = JobLib.load('Models/Scaler_SVC.pkl')

@App.route('/')
def Home():
    # Render the homepage.
    return Render_Template('Index.html')

@App.route('/predict', methods=['POST'])
def Predict():
    # Handle prediction requests.
    if Request.method == 'POST':
        try:
            # Extract form data and transform it into a numpy array.
            Features = [float(X) for X in Request.form.values()]
            Features = NP.array(Features).reshape(1, -1)

            # Scale the features using the loaded scaler.
            Scaled_Features = Scaler.transform(Features)

            # Predict wine quality using the loaded model.
            Prediction = Model.predict(Scaled_Features)

            # Map prediction to a quality level.
            Quality = int(Prediction[0])

            # Render the homepage with prediction result.
            return Render_Template('Index.html', prediction_text=f'Predicted Wine Quality: {Quality}')

        except Exception as E:
            # Handle errors and return as JSON response.
            return JSONify({'error': str(E)})

if __name__ == '__main__':
    # Run the Flask application in debug mode.
    App.run(debug=True)
