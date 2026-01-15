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
        probs = model.predict_proba(scaler.transform(features))[0]
        pred = int(np.argmax(probs))
        return jsonify({'success': True, 'predicted_class': pred, 'confidence': f"{max(probs)*100:.1f}%"})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5003)
