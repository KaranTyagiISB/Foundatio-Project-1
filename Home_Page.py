import warnings
warnings.filterwarnings('ignore')
import streamlit as st

st.set_page_config(
    page_title="BestFynd",
    page_icon="👋",
)

from PIL import Image
image = Image.open('logo.png')
st.image(image)

st.title("BestFynd")
st.write("BestFynd, is the one stop shop for all your human resource needs. ")
