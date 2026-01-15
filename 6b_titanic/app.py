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
        d = request.get_json()
        sex_enc = 0 if d['sex'] == 'female' else 1
        emb_map = {'C': 0, 'Q': 1, 'S': 2}
        features = np.array([[d['pclass'], sex_enc, d['age'], d['sibsp'], d['parch'], d['fare'], emb_map.get(d['embarked'], 2)]])
        prob = model.predict_proba(scaler.transform(features))[0][1]
        return jsonify({'success': True, 'survived': prob >= 0.5, 'message': 'Likely to SURVIVE!' if prob >= 0.5 else 'Unlikely to survive', 'probability_percent': f"{prob*100:.1f}%"})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
