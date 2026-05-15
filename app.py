import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# 1. Modeli ve Ölçekleyiciyi Yükle
model = joblib.load('gb_model.pkl')
scaler = joblib.load('scaler.pkl')

# Sayfa Başlığı
st.set_page_config(page_title="Emlak Fiyat Tahmini", page_icon="🏠")
st.title("🏠 Gayrimenkul Değerleme Sistemi")

# 2. Kullanıcı Giriş Alanları (Senin değişkenlerin)
st.subheader("Mülk Özelliklerini Giriniz")
col1, col2 = st.columns(2)

with col1:
    h_age = st.number_input("Bina Yaşı (house_age)", value=15.0)
    dist_mrt = st.number_input("Metroya Mesafe (distance_mrt)", value=450.5)
    stores = st.number_input("Market Sayısı (convenience_stores)", value=4, step=1)

with col2:
    lat = st.number_input("Enlem (latitude)", value=24.96, format="%.5f")
    lon = st.number_input("Boylam (longitude)", value=121.53, format="%.5f")

# Hesaplama Butonu
if st.button("Fiyatı Tahmin Et ve Kaydet"):
    
    # Giriş verilerini senin formatına getiriyoruz
    input_data = np.array([[h_age, dist_mrt, stores, lat, lon]])
    
    # 3. Tahmini Gerçekleştir
    input_scaled = scaler.transform(input_data)
    tahmini_fiyat = model.predict(input_scaled)[0]
    
    # Sonucu ekrana şık bir şekilde yazdır
    st.success(f"### 🎯 Tahmin Edilen Birim Fiyat: {tahmini_fiyat:.2f}")

    # 4. Veritabanına Yazma (Senin SQLite kodun)
    db_url = "sqlite:///real_estate.db"
    engine = create_engine(db_url)
    
    ev_verisi = {
        'house_age': h_age,
        'distance_mrt': dist_mrt,
        'convenience_stores': stores,
        'latitude': lat,
        'longitude': lon
    }
    
    log_df = pd.DataFrame([ev_verisi])
    log_df['predicted_price'] = tahmini_fiyat
    log_df['prediction_time'] = pd.Timestamp.now()
    
    log_df.to_sql('predictions_log', engine, if_exists='append', index=False)
    st.info("💡 Tahmin parametreleri ve sonuç veritabanına kaydedildi.")

# 5. BONUS: Tahmin Geçmişini Göster (Senin kodun)
st.divider()
st.subheader("📜 Son 5 Tahmin Kaydı")
try:
    engine = create_engine("sqlite:///real_estate.db")
    history_df = pd.read_sql("SELECT * FROM predictions_log ORDER BY prediction_time DESC LIMIT 5", engine)
    st.table(history_df)
except:
    st.write("Henüz kayıtlı tahmin bulunmuyor.")
