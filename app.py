import streamlit as st
from utils.preprocessing import load_forest_fire_model, preprocess_image

st.set_page_config(page_title="🔥 Forest Fire Detection", layout="centered")

st.title("🌲🔥 Forest Fire Detection App")
st.write("Upload an image to check if it contains a forest fire.")

model = load_forest_fire_model()

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=False)
    
    img_array = preprocess_image(uploaded_file)
    prediction = model.predict(img_array)[0][0]
    
    if prediction > 0.5:
        st.error("⚠️ Forest Fire Detected!")
    else:
        st.success("✅ No Forest Fire Detected.")
