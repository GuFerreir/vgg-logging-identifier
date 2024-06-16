# predict_class.py
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model

# Define paths
base_dir = 'images'  # Caminho para a pasta de imagens
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# Set up data generators with resizing to 500x500
target_size = (500, 500)

# Load the trained model
model = load_model('vgg16_fine_tuned.h5')

# Function to predict the class of a test image
def predict_class(model, query_img_path, target_size=(500, 500)):
    # Pre-process the query image
    query_img = load_img(query_img_path, target_size=target_size)
    query_img_array = img_to_array(query_img)
    query_img_array = np.expand_dims(query_img_array, axis=0)
    query_img_array = preprocess_input(query_img_array)

    # Predict the class probabilities
    class_probabilities = model.predict(query_img_array)
    
    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(class_probabilities)

    # Get the class label associated with the predicted class index
    class_labels = sorted(os.listdir(train_dir))
    predicted_class_label = class_labels[predicted_class_index]

    return (f"{predicted_class_label}")

# Example usage
# query_image_path = 'images/test/class1-deforestation/test_16.jpg'
# predicted_class = predict_class(model, query_image_path)
# print(predicted_class)