import joblib 
import streamlit as st
import pandas as pd
import numpy as np

st.title('Spotify Music Mood Classifier')
st.markdown('Predict song mood based on audio features using machine learning')

@st.cache_resource
def load_models():
    try:
        models = joblib.load('models.pkl')
        scaler = joblib.load('scaler.pkl')
        return models, scaler
    except Exception as e:
        st.error(f'Error loading models: {str(e)}')
        st.stop()

models, scaler = load_models()

labels = {0: 'Sad', 1: 'Happy', 2: 'Energetic', 3: 'Calm'}

model_options = list(i for i, j in models)
selected_model = st.sidebar.selectbox('Select Model', model_options, index=model_options.index('Random Forest'))

st.sidebar.markdown("---")
st.sidebar.markdown('### Model Performance')
st.sidebar.markdown(f'''
**Random Forest**: 97.6% accuracy  
**LightGBM**: 96.9% accuracy  
**Decision Tree**: 96.0% accuracy  
**Logistic Regression**: 84.7% accuracy  
**SVC**: 81.7% accuracy ''')

st.markdown('### Enter song features')

col1, col2 = st.columns(2)
with col1:
    duration = st.number_input("Duration (ms)", min_value=0.0, max_value=3000000.0, value=200000.0, help="Song length in milliseconds")
    danceability = st.slider("Danceability", 0.0, 1.0, 0.5, help="How suitable the track is for dancing (0-1)")
    energy = st.slider("Energy", 0.0, 1.0, 0.5, help="Perceived intensity and activity (0-1)")
    loudness = st.slider("Loudness (dB)", -60.0, 5.0, -5.0, help="Overall loudness in decibels")
    speechiness = st.slider("Speechiness", 0.0, 1.0, 0.1, help="Presence of spoken words (0-1)")
    acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5, help="Likelihood the track is acoustic (0-1)")

with col2:
    instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.5, help="Likelihood of no vocals (0-1)")
    liveness = st.slider("Liveness", 0.0, 1.0, 0.1, help="Presence of a live audience (0-1)")
    valence = st.slider("Valence", 0.0, 1.0, 0.5, help="Musical positivity conveyed (0-1)")
    tempo = st.number_input("Tempo (BPM)", min_value=0.0, max_value=250.0, value=120.0, help="Beats per minute")
    spec_rate = st.number_input("Spectral Rate", min_value=0.0, max_value=0.00001, value=0.0000003, format="%.10f", help="Spectral audio rate feature")

if st.button("🎯 Predict Mood", type="primary"):
    inputs = np.array([[duration, danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, spec_rate]])

    model = models[model_options.index()]
    
    scaled_inputs = scaler.transform(inputs)


    prediction = model.predict(scaled_inputs)[0]

    predicted_mood = labels[prediction]

    st.markdown("---")
    st.markdown("### 🎭 Prediction Result")

    col_result1, col_result2, col_result3 = st.columns([1, 2, 1])
    with col_result2:
        st.markdown(f"<h1 style='text-align: center; color: #1DB954;'>{predicted_mood}</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>Model: {selected_model}</p>", unsafe_allow_html=True)

    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(scaled_inputs)[0]
        st.markdown("### Confidence Scores")

        confidence_df = pd.DataFrame({
            'Mood': [labels[i] for i in range(4)],
            'Confidence': [f"{p*100:.1f}%" for p in proba]})


st.markdown("---")
st.markdown("### 💡 Feature Examples")



