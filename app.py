from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Model ve Scaler dosyalarını yüklüyoruz
model = joblib.load('gb_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return "Arsa Fiyat Tahmin Uygulaması Canlıda!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Link üzerinden gelen verileri al (JSON formatında)
        data = request.get_json()
        
        # Veriyi DataFrame'e dönüştür
        df = pd.DataFrame([data])
        
        # Ölçeklendirme (Scaler) uygula
        df_scaled = scaler.transform(df)
        
        # Tahmin yap
        prediction = model.predict(df_scaled)
        
        return jsonify({'tahmin_fiyat': float(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    # Render için port ayarı
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
