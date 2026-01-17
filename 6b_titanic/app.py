from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pickle
import os

app = Flask(__name__)
CORS(app)

model_path = os.path.join(os.path.dirname(__file__), 'model')
model = pickle.load(open(os.path.join(model_path, 'titanic_survival_model.pkl'), 'rb'))
scaler = pickle.load(open(os.path.join(model_path, 'scaler.pkl'), 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        d = request.get_json()
        sex_enc = 0 if d['sex'] == 'female' else 1
        features = np.array([[d['pclass'], sex_enc, d['age'], d['sibsp'], d['fare']]])
        prob = model.predict_proba(scaler.transform(features))[0][1]
        survived = prob >= 0.5
        return jsonify({
            'success': True, 
            'survived': survived, 
            'message': 'Survived!' if survived else 'Did Not Survive',
            'probability': f"{prob*100:.1f}%"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
