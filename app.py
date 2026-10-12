"""Streamlit interface for trained photometric redshift models."""

import math
from pathlib import Path

import streamlit as st

from src.inference.predictor import RedshiftPredictor
from src.training.common import load_config


st.set_page_config(page_title="Photometric Redshift Estimation", page_icon="🌌")
st.title("Photometric Redshift Estimation")
st.write(
    "Estimate a galaxy's redshift using SDSS photometry, a Galaxy Zoo image, "
    "or a trained multimodal model."
)
st.info(
    "Predictions are produced only by trained checkpoints. This interface does "
    "not generate placeholder or fabricated estimates."
)

config_path = Path("configs/config.yaml")
try:
    config = load_config(config_path)
except (FileNotFoundError, ValueError, KeyError) as exc:
    st.error(f"Could not load project configuration: {exc}")
    st.stop()

model_name = st.selectbox(
    "Model",
    ["Photometric-only", "Image-only", "Hybrid"],
    help="Hybrid predictions require both the image and every configured photometric feature.",
)
needs_photo = model_name in ("Photometric-only", "Hybrid")
needs_image = model_name in ("Image-only", "Hybrid")

uploaded_image = None
features = {}
if needs_image:
    uploaded_image = st.file_uploader(
        "Upload a galaxy image", type=["png", "jpg", "jpeg", "tif", "tiff"]
    )
if needs_photo:
    st.subheader("Photometric features")
    st.caption(
        "Enter values in the same units and catalogue convention used during training."
    )
    columns = config["data"]["photometric_feature_columns"]
    with st.form("photometric_features"):
        for column in columns:
            features[column] = st.text_input(column, value="", key=f"feature_{column}")
        submitted = st.form_submit_button("Predict redshift")
else:
    submitted = st.button("Predict redshift")

st.subheader("Model information and limitations")
st.write(
    "The selected checkpoint must be trained on the same feature schema, image "
    "preprocessing, and catalogue conventions as the submitted inputs. A prediction "
    "is not a substitute for scientific validation. Performance depends on training "
    "sample coverage, redshift labels, image matching quality, and domain shift."
)

if submitted:
    if needs_image and uploaded_image is None:
        st.error("Please upload a galaxy image for the selected model.")
    else:
        try:
            numeric_features = None
            if needs_photo:
                numeric_features = {}
                for column, value in features.items():
                    try:
                        parsed = float(value)
                    except ValueError as exc:
                        raise ValueError(
                            f"Enter a finite numeric value for '{column}'."
                        ) from exc
                    if not math.isfinite(parsed):
                        raise ValueError(
                            f"Enter a finite numeric value for '{column}'."
                        )
                    numeric_features[column] = parsed
            predictor = RedshiftPredictor(config)
            image = None
            if uploaded_image is not None:
                from PIL import Image

                image = Image.open(uploaded_image)
            redshift = predictor.predict(
                model_name,
                photometric_features=numeric_features,
                image=image,
            )
            st.success(f"Estimated redshift: **{redshift:.6f}**")
        except (FileNotFoundError, ImportError, TypeError, ValueError) as exc:
            st.error(str(exc))
else:
    checkpoint_key = {
        "Photometric-only": "photometric_checkpoint",
        "Image-only": "image_checkpoint",
        "Hybrid": "multimodal_checkpoint",
    }[model_name]
    checkpoint = Path(config["models"][checkpoint_key])
    if not checkpoint.is_file():
        st.warning(
            f"No trained checkpoint is available for {model_name} at `{checkpoint}`. "
            "Prepare and verify datasets, then run the matching training command."
        )
