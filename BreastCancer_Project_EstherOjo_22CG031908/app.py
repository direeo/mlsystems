from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pickle
import os

app = Flask(__name__)
CORS(app)

model_path = os.path.join(os.path.dirname(__file__), 'model')
model = pickle.load(open(os.path.join(model_path, 'breast_cancer_model.pkl'), 'rb'))
scaler = pickle.load(open(os.path.join(model_path, 'scaler.pkl'), 'rb'))

FEATURES = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        prob = model.predict_proba(scaler.transform(features))[0]
        is_malignant = prob[1] >= 0.5
        return jsonify({
            'success': True,
            'prediction': 'Malignant' if is_malignant else 'Benign',
            'is_malignant': is_malignant,
            'confidence': f"{max(prob)*100:.1f}%"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5002)
