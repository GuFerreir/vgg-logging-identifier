import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.metrics.pairwise import cosine_similarity

# Define paths
base_dir = 'images'  # Caminho para a pasta de imagens
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# Set up data generators with resizing to 125x125
target_size = (125, 125)
train_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=preprocess_input)
validation_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=target_size,  # Redimensionar para 125x125 pixels
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=target_size,  # Redimensionar para 125x125 pixels
    batch_size=32,
    class_mode='categorical'
)

# Load the VGG16 model pre-trained on ImageNet
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(125, 125, 3))

# Add custom top layers for our classifier
x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the VGG16 model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Save the trained model in native Keras format (.h5)
model.save('vgg16_fine_tuned.h5')

# Load the trained model
model = load_model('vgg16_fine_tuned.h5')

# Function to find if the test image belongs to class 1 of the training set
def test_image_belongs_to_class1(model, query_img_path, data_dir):
    # Pre-process the query image
    query_img = load_img(query_img_path, target_size=target_size)
    query_img_array = img_to_array(query_img)
    query_img_array = np.expand_dims(query_img_array, axis=0)
    query_img_array = preprocess_input(query_img_array)

    # Extract features from the query image
    feature_extractor = Model(inputs=model.input, outputs=model.layers[-2].output)
    query_features = feature_extractor.predict(query_img_array)

    similarities = []
    # Iterate over all images in the dataset directory
    for subdir, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png')):
                img_path = os.path.join(subdir, file)
                img = load_img(img_path, target_size=target_size)
                img_array = img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)
                features = feature_extractor.predict(img_array)
                similarity = cosine_similarity(query_features, features)
                similarities.append(similarity[0][0])

    # Calculate the average similarity
    avg_similarity = np.mean(similarities)

    # Check if the average similarity is greater than 0.55
    if avg_similarity > 0.7:
        print("ALTA probabilidade de apresentar desmatamento (" + str(avg_similarity)+ ")")
    else:
        print("BAIXA probabilidade de apresentar desmatamento (" + str(avg_similarity)+ ")")

# Example usage
query_image_path = 'images/test/class1/test_135.jpg'
test_image_belongs_to_class1(model, query_image_path, train_dir)