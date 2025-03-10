from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Load your trained model
model = joblib.load('gradient_boosting_model.joblib')

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_result = None
    error_message = None

    if request.method == 'POST':
        try:
            # Get data from form
            data = {
                'age': float(request.form['age']),
                'sex': float(request.form['sex']),
                'cp': float(request.form['cp']),
                'trestbps': float(request.form['trestbps']),
                'chol': float(request.form['chol']),
                'fbs': float(request.form['fbs']),
                'restecg': float(request.form['restecg']),
                'thalach': float(request.form['thalach']),
                'exang': float(request.form['exang']),
                'oldpeak': float(request.form['oldpeak']),
                'slope': float(request.form['slope']),
                'ca': float(request.form['ca']),
                'thal': float(request.form['thal'])
            }
            logging.debug(f"Received data: {data}")
            features = pd.DataFrame([data])
            
            prediction = model.predict_proba(features)[0]
            logging.debug(f"Model prediction: {prediction}")
            has_disease = prediction[1] > 0.5
            severity = prediction[1] * 100

            prediction_result = {
                'prediction': 'Yes' if has_disease else 'No',
                'severity': f'{severity:.2f}%' if has_disease else 'N/A'
            }

        except Exception as e:
            error_message = str(e)
            logging.error(f"Error occurred: {error_message}")

    return render_template('index.html', prediction_result=prediction_result, error_message=error_message)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        logging.debug(f"Received JSON data: {data}")
        features = pd.DataFrame([data])
        prediction = model.predict_proba(features)[0]
        has_disease = prediction[1] > 0.5
        severity = prediction[1] * 100

        return jsonify({
            'prediction': 'Yes' if has_disease else 'No',
            'severity': f'{severity:.2f}%' if has_disease else 'N/A'
        })
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
