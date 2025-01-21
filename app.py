from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

with open('/home/redsat/Téléchargements/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('/home/redsat/Téléchargements/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json() #request data 
        features = np.array(data['features']).reshape(1, -1)
        features_scaled = scaler.transform(features) #scaling 
        prediction = model.predict(features_scaled) #inference 
        
        return jsonify({
            'success': True,
            'prediction': int(prediction[0] > 0.5),
            'probability': float(prediction[0])
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)