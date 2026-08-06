import streamlit as st
import numpy as np
import librosa
import librosa.display
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Page configuration
st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom UI Styling
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #ff2b2b;
    }
    </style>
""", unsafe_allow_html=True)

# Define Genre Classes
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Model Loading with Error Handling
@st.cache_resource
def load_cortex_model():
    model_path = 'best_genre_model.keras'
    
    if not os.path.exists(model_path):
        st.error(f"❌ Critical Error: Model file not found at `{model_path}`. Please ensure it is uploaded to the correct directory.")
        return None
        
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model architecture/weights: {e}")
        return None

model = load_cortex_model()

# Validate model loading
if model is None:
    st.stop()

# UI Layout Header
st.title("🎵 Music Genre Classification Engine")
st.markdown("Upload a full-length audio track (e.g., 30-second WAV/MP3) to analyze its features using a production-ready CNN architecture.")

# File Uploader
uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3", "ogg", "flac"])

if uploaded_file is not None:
    try:
        # Display audio player
        st.audio(uploaded_file, format='audio/wav')
        
        # Audio Loading
        with st.spinner("🔄 Loading and decoding audio file..."):
            audio, sr = librosa.load(
                uploaded_file,
                sr=22050,
                mono=True,
                duration=30
            )
            
        duration = librosa.get_duration(y=audio, sr=sr)
        
        # Validate Audio Duration
        if duration < 25:
            st.error("Please upload an audio clip close to 30 seconds.")
            st.stop()
            
        # Track Metadata Cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="📁 File Name", value=uploaded_file.name[:15] + "..." if len(uploaded_file.name) > 15 else uploaded_file.name)
        with col2:
            st.metric(label="⏱ Duration", value=f"{duration:.2f} sec")
        with col3:
            st.metric(label="🎼 Sample Rate", value=f"{sr} Hz")
            
        # Mel Spectrogram Parameters
        n_mels = 128
        n_fft = 2048
        hop_length = 512
        fmax = 8000
        
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=sr, 
            n_mels=n_mels, 
            n_fft=n_fft, 
            hop_length=hop_length, 
            fmax=fmax
        )
        
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Spectrogram Size & Padding/Cropping Logic (Target Width: 1280)
        target_width = 1280
        current_width = log_mel_spec.shape[1]
        
        if current_width < target_width:
            pad_width = target_width - current_width
            log_mel_spec = np.pad(log_mel_spec, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            log_mel_spec = log_mel_spec[:, :target_width]
            
        # Show Mel Spectrogram before prediction
        st.markdown("### Generated Mel Spectrogram")
        fig_spec, ax_spec = plt.subplots(figsize=(10, 4))
        fig_spec.patch.set_facecolor('#0e1117')
        ax_spec.set_facecolor('#0e1117')
        img = librosa.display.specshow(log_mel_spec, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', ax=ax_spec, cmap='coolwarm')
        fig_spec.colorbar(img, ax=ax_spec, format='%+2.0f dB')
        ax_spec.set_title('Mel-Spectrogram', color='white', fontsize=12)
        ax_spec.tick_params(colors='white', labelsize=8)
        ax_spec.xaxis.label.set_color('white')
        ax_spec.yaxis.label.set_color('white')
        st.pyplot(fig_spec)
            
        if st.button('Run Deep Learning Inference'):
            with st.spinner('🎼 Preprocessing Pipeline & Executing model prediction...'):
                # Reshape for CNN input: (Batch, Height, Width, Channels) -> (1, 128, 1280, 1)
                X_input = np.expand_dims(log_mel_spec, axis=-1)
                X_input = np.expand_dims(X_input, axis=0)
            
                # Prediction
                preds = model.predict(X_input)[0]
                
                # Sort top predictions
                top_indices = np.argsort(preds)[::-1]
                
                top_prediction = GENRES[top_indices[0]]
                top_confidence = float(preds[top_indices[0]]) * 100
                
                # Low Confidence Warning
                if top_confidence < 40:
                    st.warning("⚠️ The model is not confident about this prediction. The uploaded audio may not clearly belong to a single genre.")
                
                # Results Display
                st.markdown("---")
                res_col1, res_col2 = st.columns([1, 1])
                with res_col1:
                    st.markdown(f"### 🏆 Top Prediction")
                    st.success(f"**{top_prediction.upper()}**")
                with res_col2:
                    st.markdown(f"### 📈 Confidence")
                    st.info(f"**{top_confidence:.2f}%**")
                    
                st.markdown("### 🥇 Top-3 Genres Ranked")
                for i in range(3):
                    idx = top_indices[i]
                    genre_name = GENRES[idx]
                    score = float(preds[idx]) * 100
                    st.write(f"**{i+1}. {genre_name.capitalize()}** — {score:.2f}%")
                    
                # Probability Chart
                st.markdown("### 📊 Full Probability Distribution")
                
                fig, ax = plt.subplots(figsize=(10, 5))
                fig.patch.set_facecolor('#0e1117')
                ax.set_facecolor('#0e1117')
                
                sorted_indices = np.argsort(preds)
                sorted_genres = [GENRES[i].capitalize() for i in sorted_indices]
                sorted_preds = [float(preds[i]) * 100 for i in sorted_indices]
                
                bars = ax.barh(sorted_genres, sorted_preds, color='#ff4b4b')
                
                ax.set_xlabel('Confidence (%)', color='white', fontsize=12)
                ax.set_title('Genre Class Probabilities', color='white', fontsize=14, fontweight='bold')
                ax.tick_params(colors='white', labelsize=10)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_color('white')
                ax.spines['left'].set_color('white')
                
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                            va='center', ha='left', color='white', fontsize=9)
                            
                plt.tight_layout()
                st.pyplot(fig)
                
    except Exception as e:
        st.error(f"⚠️ An error occurred during audio file decoding or pipeline processing: {str(e)}")
        st.warning("Please ensure the uploaded file is a valid, uncorrupted audio format (WAV, MP3, etc.).")
