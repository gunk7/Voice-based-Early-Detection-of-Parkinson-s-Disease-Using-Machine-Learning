import streamlit as st
import librosa
import numpy as np
import torch
from transformers import WavLMModel, AutoFeatureExtractor
from spectrum import aryule
import joblib

# -------------------------------
# Load Model & Scaler
# -------------------------------
model = joblib.load("hybrid_rf_20251105_1145.joblib")
scaler = joblib.load("hybrid_scaler.joblib")
pca = joblib.load("hybrid_pca_65.joblib")  # load the PCA used in training

# -------------------------------
# WavLM setup
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
wavlm_model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(device)

# -------------------------------
# Feature extraction functions
# -------------------------------
def extract_wavlm_embedding(file_obj):
    file_obj.seek(0)
    audio, sr = librosa.load(file_obj, sr=16000)
    inputs = extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = wavlm_model(inputs.input_values.to(device))
        emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return emb.squeeze()

def extract_handcrafted_features(y, sr, lpc_order=12, mfcc_n=12):
    # LPC
    try:
        lpc_coeffs, _ = aryule(y, lpc_order)
        lpc_feat = np.concatenate([lpc_coeffs, np.full(lpc_order, np.var(lpc_coeffs))])
    except:
        lpc_feat = np.zeros(lpc_order*2)

    # LAR
    try:
        lar = np.log(np.abs(lpc_coeffs) + 1e-6)
        lar_feat = np.concatenate([lar, np.full(lpc_order, np.var(lar))])
    except:
        lar_feat = np.zeros(lpc_order*2)

    # MFCC
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=mfcc_n)
        mfcc_feat = np.concatenate([np.mean(mfcc, axis=1), np.var(mfcc, axis=1)])
    except:
        mfcc_feat = np.zeros(mfcc_n*2)

    # Combine handcrafted features
    features = np.concatenate([lpc_feat, lar_feat, mfcc_feat])
    return features

def predict_hybrid(file_obj):
    file_obj.seek(0)
    y, sr = librosa.load(file_obj, sr=None)
    handcrafted_features = extract_handcrafted_features(y, sr)

    file_obj.seek(0)
    wavlm_features = extract_wavlm_embedding(file_obj)

    # ---- Apply PCA first ----
    wavlm_reduced = pca.transform(wavlm_features.reshape(1, -1))

    # ---- Combine & scale ----
    combined_features = np.hstack([handcrafted_features.reshape(1, -1), wavlm_reduced])
    if combined_features.shape[1] != scaler.mean_.shape[0]:
        raise ValueError(f"Feature mismatch: got {combined_features.shape[1]}, expected {scaler.mean_.shape[0]}")
    
    scaled_features = scaler.transform(combined_features)
    pred_label = model.predict(scaled_features)[0]
    pred_prob = model.predict_proba(scaled_features)[0]
    return pred_label, pred_prob

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Parkinson's Voice Classifier", page_icon="🧠", layout="centered")
st.title("🧠 Parkinson’s Disease Voice Classifier")
st.markdown("Upload a `.wav` file to predict whether the speaker shows Parkinsonian traits.")

uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    with st.spinner('Extracting features and predicting...'):
        try:
            label, prob = predict_hybrid(uploaded_file)
            st.write(f"## Prediction: **{label}**")
            st.write("### Class Probabilities")
            st.bar_chart(prob)
        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")
