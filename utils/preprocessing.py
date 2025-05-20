from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

def load_forest_fire_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "model", "FFD.keras")
    print("Attempting to load model from:", model_path)
    
    return load_model(model_path)



def preprocess_image(uploaded_file):
    img = image.load_img(uploaded_file, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

