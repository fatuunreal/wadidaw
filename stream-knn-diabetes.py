import pickle
import streamlit as st

# membaca model
diabetes_model = pickle.load(open('knn_diabetes_model.sav', 'rb'))

#judul web
st.title('Prediksi Diabetes Dengan KNN wadidaw')

#membagi kolom
col1, col2 = st.columns(2)

with col1:
    Pregnancies = st.text_input('Input nilai Pregnancies', '0')
with col2:
    Glucose = st.text_input('Input nilai Glucose', '0')

with col1:
    BloodPressure = st.text_input('Input nilai Blood Pressure', '0')
with col2:
    SkinThickness = st.text_input('Input nilai Skin Thickness', '0')

with col1:
    Insulin = st.text_input('Input nilai Insulin', '0')
with col2:
    BMI = st.text_input('Input nilai BMI', '0')

with col1:
    DiabetesPedigreeFunction = st.text_input('Input nilai Diabetes Pedigree Function', '0')
with col2:
    Age = st.text_input('Input nilai Age', '0')

# code untuk prediksi
diab_diagnosis = ''

# membuat tombol untuk prediksi
if st.button('Test Prediksi Diabetes'):
    try:
        # Konversi input ke float
        inputs = [float(Pregnancies), float(Glucose), float(BloodPressure),
                  float(SkinThickness), float(Insulin), float(BMI),
                  float(DiabetesPedigreeFunction), float(Age)]
        
        # Prediksi
        diab_prediction = diabetes_model.predict([inputs])
        
        if diab_prediction[0] == 1:
            diab_diagnosis = 'Pasien terkena Diabetes'
        else:
            diab_diagnosis = 'Pasien tidak terkena Diabetes'
    except ValueError:
        diab_diagnosis = 'Masukkan semua input sebagai angka valid!'

st.success(diab_diagnosis)

# Menampilkan metrik model
st.title("Evaluasi Model KNN untuk Prediksi Diabetes")

# Menampilkan metrik utama
st.subheader("Metrik Evaluasi")
st.metric(label="Akurasi", value="92.68%")
st.metric(label="Presisi", value="94.75%")
st.metric(label="Recall (Sensitivitas)", value="93.53%")
st.metric(label="ROC AUC", value="98.33%")

# Penjelasan tambahan (opsional)
st.write("""
- **Akurasi**: Proporsi prediksi yang benar secara keseluruhan.
- **Presisi**: Proporsi prediksi positif yang benar-benar positif.
- **Recall**: Kemampuan model mendeteksi kasus positif.
- **ROC AUC**: Gambaran umum kinerja model di berbagai ambang prediksi.
""")

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Contoh data (ganti dengan data ROC Anda)
fpr = [0.0, 0.1, 0.2, 1.0]
tpr = [0.0, 0.8, 0.9, 1.0]
roc_auc = 0.9833

# Membuat plot ROC
fig, ax = plt.subplots()
ax.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend()

# Menampilkan di Streamlit
st.pyplot(fig)

