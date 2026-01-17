from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pickle
import os

app = Flask(__name__)
CORS(app)

# Load model and scaler
model_path = os.path.join(os.path.dirname(__file__), 'model')
model = pickle.load(open(os.path.join(model_path, 'house_price_model.pkl'), 'rb'))
scaler = pickle.load(open(os.path.join(model_path, 'scaler.pkl'), 'rb'))

FEATURES = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'BedroomAbvGr', 'YearBuilt']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        price = float(prediction[0])
        return jsonify({'success': True, 'predicted_price': price, 'formatted_price': f"${price:,.2f}"})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
