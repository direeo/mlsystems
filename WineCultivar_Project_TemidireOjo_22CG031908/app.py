from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pickle
import os

app = Flask(__name__)
CORS(app)

model_path = os.path.join(os.path.dirname(__file__), 'model')
model = pickle.load(open(os.path.join(model_path, 'wine_cultivar_model.pkl'), 'rb'))
scaler = pickle.load(open(os.path.join(model_path, 'scaler.pkl'), 'rb'))

FEATURES = ['alcohol', 'malic_acid', 'ash', 'magnesium', 'flavanoids', 'color_intensity']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        probs = model.predict_proba(scaler.transform(features))[0]
        pred = int(np.argmax(probs))
        return jsonify({
            'success': True,
            'predicted_class': pred,
            'cultivar': f"Cultivar {pred + 1}",
            'confidence': f"{max(probs)*100:.1f}%"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5003)
