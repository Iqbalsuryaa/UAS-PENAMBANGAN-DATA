import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from streamlit_option_menu import option_menu

# Navigasi sidebar
# Horizontal menu
selected2 = option_menu(
    None,
    ["Data", "Processing data", "Modeling", "Implementasi"],
    icons=["house", "cloud-upload", "list-task", "gear"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal"
)

# Halaman Data
if selected2 == "Data":
    st.title("Deskripsi data")

    st.write("Ini adalah contoh data yang tersedia dalam aplikasi Streamlit.")
    st.write("Data ini berisikan informasi mengenai pria dan wanita, yang bertujuan untuk mengetahui memiliki rambut yang panjang atau tidak.")
    st.write("Penjelasan Tipe Data : Gender : wanita atau pria, yang Merupakan type data Kategorial yang nantinya akan kita ubah ke type data biner, dengan nilai value 1 untuk wanita dan value 0 untuk pria. dahi_lebar_cm : lebar dahi dari kanan ke kiri diberikan dalam ukuran cm. Merupakan type data Numeric dahi_tinggi_cm : lebar dahi lebar dalam ukuran cm dari tempat rambut tumbuh ke alis. Merupakan type data Numeric. Apakah memiliki hidung yang lebar atau tidak : 1 mewakili lebar dan 0 tidak. Merupakan Type data biner. Apakah memiliki hidung yang panjang atau tidak : 1 mewakili panjang dan 0 tidak. Merupakan Type data biner. Apakah orang ini memiliki bibir yang tipis atau tidak : 1 mewakili kurus dan 0 tidak. Apakah jarak dari hidung ke bibir panjang : 1 mewakili ya dan 0 tidak. Menunjukkan apakah orang tersebut berambut panjang atau tidak : 1 adalah rambut panjang dan 0 adalah rambut tidak panjang")
    st.write("Data ini diambil dari kaggle")
    data = pd.read_csv('gender_classification_v7_oke.csv')
    st.write(data)

# Halaman Processing data
if selected2 == "Processing data":
    st.title("Processing Data")

    st.write("Saya menggunakan processing data SKALASI STANDAR.")
    st.write("Dengan hasil processing data")
    data = pd.read_csv('preprocessed_data.csv')
    st.write(data)

# Halaman Modeling
if selected2 == "Modeling":
    st.title("Modeling")

    pilih = st.radio("Pilih", ("Naive Bayes", "Decision Tree", "KNN", "MLP"))

    if pilih == "Naive Bayes":
        st.title(" Nilai Akurasi 88%")
    elif pilih == "Decision Tree":
        st.title(" Nilai Akurasi 87%")
    elif pilih == "KNN":
        st.title(" Nilai Akurasi 85%")
    elif pilih == "MLP":
        st.title(" Nilai Akurasi 87%")

# Halaman Implementasi
if selected2 == "Implementasi":
    st.title("Implementasi")

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

    main()
