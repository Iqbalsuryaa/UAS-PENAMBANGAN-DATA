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
    if (selected2 == 'Implementasi'):
    st.title('Implementasi')
          
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Load data
df = pd.read_csv("gender_classification_v7_oke.csv")

# Preprocessing data
Gen_wanita = "Female"
Gen_pria = "Male"
value_one = 1
value_zero = 0

def change_gender_to_binary(gender):
    return value_one if gender == Gen_pria else value_zero

# Update kolom 'gender'
df["gender"] = df["gender"].apply(change_gender_to_binary)

X = df.iloc[:, :6].values
y = df['long_hair'].values

sc = StandardScaler()
X = sc.fit_transform(X)

# Split data menjadi train set dan test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# Model Naive Bayes
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Fungsi untuk memprediksi hair length
def predict_hair_length(gender, forehead_width_cm, forehead_height_cm, nose_wide, nose_long, lips_thin, distance_nose_to_lip_long):
    # Preprocessing pada input data
    input_data = np.array([[gender, forehead_width_cm, forehead_height_cm, nose_wide, nose_long, lips_thin, distance_nose_to_lip_long]])
    input_data = sc.transform(input_data)
    
    # Prediksi menggunakan model Naive Bayes
    predicted_hair_length = classifier.predict(input_data)
    
    return predicted_hair_length[0]

# Halaman Streamlit
def main():
    st.title("Memprediksi seseorang tersebut berambut panjang atau tidak")

    # Input pengguna
    gender = st.selectbox("Jenis Kelamin", ["Pria", "Wanita"])
    forehead_width_cm = st.number_input("Lebar Dahi (cm)", min_value=0.0)
    forehead_height_cm = st.number_input("Tinggi Dahi (cm)", min_value=0.0)
    nose_wide = st.radio("Lebar Hidung", [0, 1])
    nose_long = st.radio("Panjang Hidung", [0, 1])
    lips_thin = st.radio("Tipisnya Bibir", [0, 1])
    distance_nose_to_lip_long = st.radio("Jarak Hidung ke Bibir (Panjang)", [0, 1])

    if st.button("Prediksi"):
        long_hair = predict_hair_length(gender, forehead_width_cm, forehead_height_cm, nose_wide, nose_long, lips_thin, distance_nose_to_lip_long)
        if long_hair == 1:
            prediction = "Berambut Panjang"
        else:
            prediction = "Tidak Berambut Panjang"
        st.write("Hasil Prediksi:", prediction)

if __name__ == "__main__":
    main()

    
    

    


    
