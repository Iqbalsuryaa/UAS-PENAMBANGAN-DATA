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

def predict_hair_length(gender, forehead_width_cm, forehead_height_cm, nose_wide, nose_long, lips_thin, distance_nose_to_lip_long):
    # Implementasikan model atau perhitungan prediksi di sini
    # Anda dapat menggunakan model machine learning, aturan sederhana, atau metode lainnya

    # Contoh sederhana: Jika lebar dahi dan tinggi dahi lebih dari 10, prediksi berambut panjang
    if forehead_width_cm > 10 and forehead_height_cm > 10:
        return "Rambut Panjang"
    else:
        return "Rambut Pendek"

# Tampilan aplikasi Streamlit
def main():
    st.title("Memprediksi Seseorang Berambut Panjang atau Tidak")
    st.write("Masukkan informasi berikut:")

    # Input pengguna
    gender = st.selectbox("Jenis Kelamin", ("Pria", "Wanita"))
    forehead_width_cm = st.number_input("Lebar Dahi (cm)")
    forehead_height_cm = st.number_input("Tinggi Dahi (cm)")
    nose_wide = st.number_input("Lebar Hidung (cm)")
    nose_long = st.number_input("Panjang Hidung (cm)")
    lips_thin = st.number_input("Ketebalan Bibir (cm)")
    distance_nose_to_lip_long = st.number_input("Jarak Antara Hidung dan Bibir (cm)")

    # Prediksi ketika tombol ditekan
    if st.button("Prediksi"):
        result = predict_hair_length(gender, forehead_width_cm, forehead_height_cm, nose_wide, nose_long, lips_thin, distance_nose_to_lip_long)
        st.write("Hasil Prediksi:", result)

if __name__ == "__main__":
    main()

    
