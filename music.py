import streamlit as st
import pandas as pd
#import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')


st.title('Music Sentiment')
st.subheader('Built by Ifeyinwa')

music = pd.read_csv('music_sentiment_dataset.csv')

st.markdown("<h1 style = 'color: #DD5746; text-align: center; font-size: 60px; font-family: Monospace'>MUSIC SENTIMENT PREDICTOR</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #FFC470; text-align: center; font-family: Serif '>Built by IFEYINWA</h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html=True)


st.image('pngwing.com.png')
st.divider()


st.markdown("<h2 style = 'color: #F7C566; text-align: center; font-family: montserrat '>Background Of Study</h2>", unsafe_allow_html = True)
st.markdown("In the rapidly evolving music industry, understanding how listeners emotionally respond to songs is crucial for personalized recommendations, targeted marketing, and mood-based playlist curation. However, accurately predicting the sentiment or emotional tone of music — whether a song is happy, sad, energetic, or calm — remains a challenging task due to the subjective nature of emotions and the complexity of musical features. This project aims to develop a system that can automatically predict the sentiment of a song based on its audio features, lyrics, and metadata, enabling more intuitive and emotion-aware music experiences for listeners.")

st.divider()

st.dataframe(music,use_container_width= True)

st.sidebar.image('music user icon.png',caption = "Welcome User")

#['Artist','Song_Name','Tempo_BPM','Mood','Genre','Recommended_Song_ID','User_Text','Sentiment_Label']

artist= st.sidebar.selectbox('Artist exp', music.Artist.unique(), index =1)
song = st.sidebar.selectbox('Song exp',music.Song_Name.unique(), index =1)
temp= st.sidebar.number_input('Tempo exp', min_value=0.0, max_value=1000.0, value=music.Tempo_BPM.median())
mood= st.sidebar.selectbox('Mood exp', music.Mood.unique(), index =1)
genre = st.sidebar.selectbox('genre exp', music.Genre.unique(),index=1)
recommended = st.sidebar.selectbox('recommended exp',music.Recommended_Song_ID.unique(),index=1)
user = st.sidebar.selectbox('user exp',music.User_Text.unique(),index=1)
energy = st.sidebar.selectbox('energy exp',music.Energy.unique(),index=1)
dance = st.sidebar.selectbox('dance exp',music.Danceability.unique(),index=1)

  

inputs = {

    'User_Text' : [user],
    'Recommended_Song_ID': [recommended],
    'Song_Name' : [song],
    'Artist' : [artist],    
    'Genre' : [genre],
    'Tempo_BPM' : [temp],
    'Mood' : [mood],    
    'Energy' : [energy],
    'Danceability':[dance]
   
}

 #if we want the input  to appear under the  dataset

inputVar = pd.DataFrame(inputs)
st.divider()
st.header('User Input')
st.dataframe(inputVar)


# transform the user inputs,import the transformers(scalers)

user_encoder = joblib.load('User_Text_encoder.pkl')
recommended_encoder = joblib.load('Recommended_Song_ID_encoder.pkl')
song_encoder = joblib.load('Song_Name_encoder.pkl')
artist_encoder= joblib.load('Artist_encoder.pkl')
genre_encoder = joblib.load('Genre_encoder.pkl')
temp_scaler = joblib.load('Tempo_BPM_scaler.pkl')
mood_encoder = joblib.load('Mood_encoder.pkl')
energy_encoder = joblib.load('Energy_encoder.pkl')
dance_encoder = joblib.load('Danceability_encoder.pkl')



inputVar['User_Text'] =user_encoder.transform(inputVar[['User_Text']])
inputVar['Recommended_Song_ID'] = recommended_encoder.transform(inputVar[['Recommended_Song_ID']])
inputVar['Song_Name'] = song_encoder.transform(inputVar[['Song_Name']])
inputVar['Artist'] = artist_encoder.transform(inputVar[['Artist']])
inputVar['Genre'] = genre_encoder.transform(inputVar[['Genre']])
inputVar['Tempo_BPM'] = temp_scaler.transform(inputVar[['Tempo_BPM']])
inputVar['Mood'] = mood_encoder.transform(inputVar[['Mood']])
inputVar['Energy'] =energy_encoder.transform(inputVar[['Energy']])
inputVar['Danceability'] =dance_encoder.transform(inputVar[['Danceability']])





##Bringing in the model
model = joblib.load('Musicmodel.pkl')


predictbutton = st.button('Push to Predict the persons sentiment concerning this music')
 
if predictbutton: 
    predicted = model.predict(inputVar)
    st.success(f'the Customer is feeling : {predicted} listening to this music')




















