import streamlit as st
import pandas as pd 


# Preparation model
df = pd.read_csv("2019.csv")
df.rename(index=str,columns={
    "GDP per capita" : "Economy",
    "Social support" : "Social",
    "Perceptions of corruption" : "Trust"
})
x = df.drop(["Country or region", "Freedom to make life choices", "Overall rank"], axis=1)  

# The ui
# Title
st.title("Segmentasi faktor Kebahagian sebuah Negara")
st.write('''
        ##### Nama : Achmad Fauzan Nabil
        ##### NIM : 211351002
        ##### Kelas : Malam B
''')
st.divider()
st.write('''
    Web app ini bertujuan untuk mendalami dan menganalisis hubungan yang kompleks antara berbagai faktor yang berpengaruh terhadap tingkat kebahagiaan suatu negara. Faktor-faktor yang menjadi fokus utama dalam web app ini meliputi tingkat kemakmuran ekonomi, kekuatan hubungan sosial dalam masyarakat, serta persepsi masyarakat terhadap tingkat kepercayaan kepada pemerintah, yang sering diindikasikan oleh tingkat korupsi yang dirasakan.
''')
# Infor dataset 
st.header("Isi dataset")
st.write(df)
