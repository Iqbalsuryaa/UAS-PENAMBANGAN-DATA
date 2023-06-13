import streamlit as st
import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from streamlit_option_menu import option_menu

#navigasi sidebar
# horizontal menu
selected2 = option_menu(None, ["Data", "Procecing data", "Modelling", 'Implementasi'], 
    icons=['house', 'cloud-upload', "list-task", 'gear'], 
    menu_icon="cast", default_index=0, orientation="horizontal")
selected2

#halaman Data
if (selected2 == 'Data') :
    st.title('deskripsi data')

    st.write("Ini adalah contoh data yang tersedia dalam aplikasi Streamlit.")
    st.write("Data ini berisikan informasi mengenai pria dan wanita, yang bertujuan untuk mengetahui memiliki rambut yang panjang atau tidak.")
    st.write("Data ini diambil dari kaggle")
    st.write("Data ini merupakan type data Numerik")
    data = pd.read_csv('gender_classification_v7_oke.csv')
    st.write(data)



#halaman procecing data 
if (selected2 == 'Procecing data') :
    st.title('Procecing Data')

    st.write("saya menggunakan procecing data SKALASI STANDAR ")
    st.write("Dengan hasil procecing data")
    data = pd.read_csv('preprocessed_data.csv')
    st.write(data)


#halaman modelling
if (selected2 == 'Modelling'):
    st.title('Modelling')

    pilih = st.radio('Pilih', ('Naive Bayes', 'Decision Tree', 'KNN', 'MLP'))

    if (pilih == 'Naive Bayes'):
        st.title(' Nilai Akurasi 88%')
    elif (pilih == 'Decision Tree'):
        st.title(' Nilai Akurasi 87%')
    elif (pilih == 'KNN'):
        st.title(' Nilai Akurasi 85%')
    elif (pilih == 'MLP'):
        st.title(' Nilai Akurasi 87%')


#halaman implementasi
# Load the saved model
if (selected2 == 'Implementasi'):
    st.title('Implementasi')
    
    model_filename = 'loan.pkl'
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)

    # Create a function to preprocess the input data
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

# Memuat data yang telah diproses sebelumnya
df_std = pd.read_csv('preprocessed_data.csv')

# Menyiapkan data untuk pemodelan
X = df_std[['gender', 'forehead_width', 'forehead_height_cm', 'nose_long', 'lips_thin', 'distance_nose_to_lip_long']].values
y = df_std['long_hair'].values

# Menstandarisasi fitur-fitur
sc = StandardScaler()
X = sc.fit_transform(X)

# Melatih model klasifikasi Gaussian Naive Bayes
classifier = GaussianNB()
classifier.fit(X, y)

# Fungsi untuk memprediksi rambut panjang berdasarkan input pengguna
def predict_long_hair(gender, forehead_width, forehead_height, nose_long, lips_thin, distance_nose_to_lip_long):
    input_data = [gender, forehead_width, forehead_height, nose_long, lips_thin, distance_nose_to_lip_long]
    input_data = sc.transform([input_data])
    predicted_long_hair = classifier.predict(input_data)
    return predicted_long_hair[0]

# Aplikasi web menggunakan Streamlit
def main():
    st.title("Prediksi Panjang Rambut")
    st.write("Masukkan informasi berikut untuk memprediksi apakah seseorang memiliki rambut panjang atau tidak:")

    gender = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
    forehead_width = st.slider("Lebar Dahi", min_value=0.0, max_value=1.0, step=0.01)
    forehead_height = st.slider("Tinggi Dahi (cm)", min_value=0.0, max_value=1.0, step=0.01)
    nose_long = st.slider("Panjang Hidung", min_value=0.0, max_value=1.0, step=0.01)
    lips_thin = st.slider("Ketipisan Bibir", min_value=0.0, max_value=1.0, step=0.01)
    distance_nose_to_lip_long = st.slider("Jarak dari Hidung ke Bibir", min_value=0.0, max_value=1.0, step=0.01)

    if st.button("Prediksi"):
        predicted_hair_length = predict_long_hair(gender, forehead_width, forehead_height, nose_long, lips_thin, distance_nose_to_lip_long)
        if predicted_hair_length == 1:
            st.write("Orang tersebut diprediksi memiliki rambut panjang.")
        else:
            st.write("Orang tersebut diprediksi tidak memiliki rambut panjang.")

if __name__ == '__main__':
    main()



    


    
