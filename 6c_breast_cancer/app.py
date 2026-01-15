from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)

model = pickle.load(open('model.h5', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = np.array(request.get_json()['features']).reshape(1, -1)
        prob = model.predict_proba(scaler.transform(features))[0][1]
        return jsonify({'success': True, 'prediction': 'Benign' if prob >= 0.5 else 'Malignant', 'is_benign': prob >= 0.5, 'benign_probability': prob, 'malignant_probability': 1-prob, 'confidence': f"{max(prob,1-prob)*100:.1f}%"})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5002)
