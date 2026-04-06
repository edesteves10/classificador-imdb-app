import streamlit as st
import pandas as pd
import pickle
import numpy as np
from tensorflow.keras.models import load_model

# 1. Configuração Visual
st.set_page_config(page_title="IA de Cinema", page_icon="🍿")
st.title("🎬 Analisador de Sentimentos")
st.write("Esta IA foi treinada com 50.000 críticas do IMDB para entender opiniões.")

# 2. Carregando a Inteligência
@st.cache_resource
def load_assets():
    model = load_model('modelo_sentimento.h5')
    with open('vetorizador.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    return model, tfidf

model, tfidf = load_assets()

# 3. Interface de Usuário
user_text = st.text_area("Digite sua crítica (em inglês):", "The movie was a true masterpiece!")

if st.button("Analisar"):
    # Transformar o texto do usuário em números
    vetor = tfidf.transform([user_text]).toarray()
    
    # Fazer a previsão
    prediction = model.predict(vetor)[0][0]
    
    if prediction > 0.5:
        st.success(f"🌟 POSITIVA! (Confiança: {prediction:.2%})")
    else:
        st.error(f"💀 NEGATIVA! (Confiança: {1-prediction:.2%})")
