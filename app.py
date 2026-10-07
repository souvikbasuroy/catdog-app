import io
import numpy as np
import requests
import streamlit as st
from PIL import Image
from huggingface_hub import hf_hub_download
from tensorflow import keras

REPO_ID = "Souvikbasur/catdog-mobilenetv2"
IMG_SIZE = (160, 160)

st.set_page_config(page_title="Cat vs Dog", page_icon="🐱")
st.title("🐱 cat vs dog classifier 🐶")
st.caption("mobilenetv2 transfer learning | 98.31% val accuracy")


@st.cache_resource
def load_model():
    path = hf_hub_download(REPO_ID, "catdog_model.keras")
    return keras.models.load_model(path)


def fetch_image(url):
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    r.raise_for_status()
    if len(r.content) > 10 * 1024 * 1024:
        raise ValueError("image bigger than 10 MB")
    return Image.open(io.BytesIO(r.content))


def predict(img, model):
    img = img.convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
    x = np.array(img, dtype="float32")[None]
    return float(model.predict(x, verbose=0)[0][0])


model = load_model()

tab_url, tab_up = st.tabs(["🔗 paste image link", "📁 upload image"])
img = None

with tab_url:
    url = st.text_input("image url", placeholder="https://...jpg")
    if url:
        try:
            img = fetch_image(url.strip())
        except Exception as e:
            st.error(f"could not load image from link: {e}")

with tab_up:
    f = st.file_uploader("choose image", type=["jpg", "jpeg", "png", "webp"])
    if f is not None:
        try:
            img = Image.open(f)
        except Exception as e:
            st.error(f"bad image file: {e}")

if img is not None:
    st.image(img, caption="input image", use_container_width=True)
    with st.spinner("predicting..."):
        p_dog = predict(img, model)
    label = "dog 🐶" if p_dog > 0.5 else "cat 🐱"
    conf = p_dog if p_dog > 0.5 else 1 - p_dog
    st.success(f"**{label}**  ({conf*100:.1f}% confident)")
    st.progress(1 - p_dog, text=f"cat: {(1-p_dog)*100:.1f}%")
    st.progress(p_dog, text=f"dog: {p_dog*100:.1f}%")
    if conf < 0.7:
        st.warning("low confidence, image may not be a clear cat or dog")

st.divider()
st.caption("model only knows cat vs dog, any other image is still forced into one of the two.")
