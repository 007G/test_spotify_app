import numpy as np
import pandas as pd
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler
 
# loading the trained model
pickle_in_model = open('modelForPrediction.sav', 'rb') 
classifier = pickle.load(pickle_in_model)

pickle_in_scaler = open('sandardScalar.sav', 'rb') 
std_scaler = pickle.load(pickle_in_scaler)
 

  
# defining the function which will make the prediction using the data which the user inputs 
def prediction(danceability,energy,key,loudness,speechiness,acousticness,instrumentalness,liveness,valence,tempo,duration_ms,mode):   
 
    # Pre-processing user input
    scalar = StandardScaler()    
    scaled_input = std_scaler.transform([[danceability,energy,key,loudness,speechiness,acousticness,instrumentalness,liveness,valence,tempo,duration_ms,mode]])
 
    
 
    # Making predictions 
    prediction = classifier.predict(scaled_input)
     
    if prediction == 0:
        pred = 'Not Hit'
    else:
        pred = 'Hit'
    return pred
      
  
# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:tomato;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Streamlit Billboard Hits Prediction ML App</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction  
    danceability = st.number_input("Danceability")
    energy = st.number_input("Energy")
    key = st.number_input("Key")
    loudness = st.number_input("Loudness")
    speechiness = st.number_input("Speechiness")
    acousticness = st.number_input("Acousticness")
    instrumentalness = st.number_input("Instrumentalness")
    liveness = st.number_input("Liveness")
    valence = st.number_input("Valence")
    tempo = st.number_input("Tempo")
    duration_ms = st.number_input("Duration_ms")
    mode = st.number_input("Mode") 
    
    result =""
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction(danceability,energy,key,loudness,speechiness,acousticness,instrumentalness,liveness,valence,tempo,duration_ms,mode) 
        st.success('Your song is {}'.format(result))
    
    if st.button("About"):
        st.text("Made By Gaurav Singh")
        st.text("Built with Streamlit")

     
if __name__=='__main__': 
    main()