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

# Define Genre Classes (must match training label order exactly)
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# These MUST match the values used in the training notebook
SR = 22050
SEGMENT_DURATION = 3.0
SAMPLES_PER_SEGMENT = int(SR * SEGMENT_DURATION)
N_MELS = 128
FMAX = 8000
TARGET_FRAMES = 130  # matches training: mel_spec_db[:, :130]
MAX_DURATION = 600.0  # safety cap: don't process more than 10 minutes of audio

# Optional: normalization stats saved from training (mean.npy / std.npy).
# If not present, we fall back to per-clip normalization (less accurate
# but keeps the app functional). See notes below on how to generate these.
NORM_STATS_PATH = "norm_stats.npz"


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


@st.cache_resource
def load_norm_stats():
    if os.path.exists(NORM_STATS_PATH):
        data = np.load(NORM_STATS_PATH)
        return float(data["mean"]), float(data["std"])
    return None, None


model = load_cortex_model()
saved_mean, saved_std = load_norm_stats()

# Validate model loading
if model is None:
    st.stop()

if saved_mean is None:
    st.warning(
        "⚠️ Training normalization stats (`norm_stats.npz`) not found. "
        "Falling back to per-clip normalization, which may reduce prediction accuracy. "
        "See the training notebook for how to export these stats."
    )

# UI Layout Header
st.title("🎵 Music Genre Classification Engine")
st.markdown("Upload an audio track (any length - full song works too) to analyze its features using a production-ready CNN architecture.")

# File Uploader
uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3", "ogg", "flac"])

if uploaded_file is not None:
    try:
        # Display audio player
        st.audio(uploaded_file, format='audio/wav')

        # Audio Loading (loads the full track, capped at MAX_DURATION for safety)
        with st.spinner("🔄 Loading and decoding audio file..."):
            audio, sr = librosa.load(
                uploaded_file,
                sr=SR,
                mono=True,
                duration=MAX_DURATION
            )

        duration = librosa.get_duration(y=audio, sr=sr)

        # Validate Audio Duration
        if duration < SEGMENT_DURATION:
            st.error(f"Please upload an audio clip at least {int(SEGMENT_DURATION)} seconds long.")
            st.stop()

        # Track Metadata Cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="📁 File Name", value=uploaded_file.name[:15] + "..." if len(uploaded_file.name) > 15 else uploaded_file.name)
        with col2:
            st.metric(label="⏱ Duration", value=f"{duration:.2f} sec")
        with col3:
            st.metric(label="🎼 Sample Rate", value=f"{sr} Hz")

        if duration >= MAX_DURATION - 1:
            st.info(f"ℹ️ Only the first {int(MAX_DURATION)} seconds are analyzed for very long tracks.")

        # ------------------------------------------------------------------
        # Split the ENTIRE track into consecutive 3-second segments,
        # exactly like the training pipeline (just not capped at 30s anymore)
        # ------------------------------------------------------------------
        segment_specs = []
        num_segments = int(len(audio) // SAMPLES_PER_SEGMENT)

        for d in range(num_segments):
            start_sample = int(d * SAMPLES_PER_SEGMENT)
            end_sample = int(start_sample + SAMPLES_PER_SEGMENT)
            segment = audio[start_sample:end_sample]

            if len(segment) < SAMPLES_PER_SEGMENT:
                continue

            mel_spec = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=N_MELS, fmax=FMAX)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            if mel_spec_db.shape[1] >= TARGET_FRAMES:
                mel_spec_db = mel_spec_db[:, :TARGET_FRAMES]
                segment_specs.append(mel_spec_db)

        if len(segment_specs) == 0:
            st.error("Could not extract any valid 3-second segments from this clip. Try a different file.")
            st.stop()

        segment_specs = np.array(segment_specs)  # (num_segments, 128, 130)

        # Normalize using saved training stats if available, else per-clip stats
        if saved_mean is not None:
            norm_mean, norm_std = saved_mean, saved_std
        else:
            norm_mean, norm_std = np.mean(segment_specs), np.std(segment_specs)

        segment_specs_norm = (segment_specs - norm_mean) / (norm_std + 1e-7)
        X_input = segment_specs_norm[..., np.newaxis]  # (num_segments, 128, 130, 1)

        # Show Mel Spectrogram of the first segment (for a quick visual)
        st.markdown("### Generated Mel Spectrogram (first 3-second segment)")
        fig_spec, ax_spec = plt.subplots(figsize=(10, 4))
        fig_spec.patch.set_facecolor('#0e1117')
        ax_spec.set_facecolor('#0e1117')
        img = librosa.display.specshow(segment_specs[0], sr=sr, hop_length=512, x_axis='time', y_axis='mel', ax=ax_spec, cmap='coolwarm')
        fig_spec.colorbar(img, ax=ax_spec, format='%+2.0f dB')
        ax_spec.set_title('Mel-Spectrogram', color='white', fontsize=12)
        ax_spec.tick_params(colors='white', labelsize=8)
        ax_spec.xaxis.label.set_color('white')
        ax_spec.yaxis.label.set_color('white')
        st.pyplot(fig_spec)

        if st.button('Run Deep Learning Inference'):
            with st.spinner('🎼 Preprocessing Pipeline & Executing model prediction...'):
                # Predict on every 3-second segment, then average probabilities
                segment_preds = model.predict(X_input)  # (num_segments, 10)
                preds = np.mean(segment_preds, axis=0)  # (10,)

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
