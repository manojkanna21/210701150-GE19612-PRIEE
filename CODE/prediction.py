'''import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the saved model
model = load_model("medcinal_plant_detection_model.keras")

# Define constants
image_size = (224, 224)

# Load and preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=image_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.  # Normalize pixel values
    return img_array

# Classify the image
def classify_image(img_array):
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    return predicted_class

# Map indices to class labels
def get_class_label(predicted_class, class_labels):
    predicted_label = class_labels[predicted_class]
    return predicted_label

# Data augmentation settings for loading class labels
data_augmentation = ImageDataGenerator(rescale=1./255)

# Dummy values for demonstration (replace with actual paths and values)
folder_path = 'data'
batch_size = 32
seed = 42

# Create a generator to load class labels
train_generator = data_augmentation.flow_from_directory(
    folder_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    seed=seed
)

# Obtain class labels
class_labels = train_generator.class_indices
class_labels = dict((v, k) for k, v in class_labels.items())

# Provide input image path
input_img_path = r"4524.jpg"

# Preprocess input image
input_img_array = preprocess_image(input_img_path)

# Classify input image
predicted_class = classify_image(input_img_array)

# Get class label
predicted_label = get_class_label(predicted_class, class_labels)

print("Predicted class:", predicted_label)'''


from flask import Flask, render_template, jsonify, request
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the saved model
model = load_model("medcinal_plant_detection_model.keras")

# Define constants
image_size = (224, 224)

# Load and preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=image_size)
    img_array = image.img_to_array(img)
    img_array = img_array.reshape((1,) + img_array.shape)  # Reshape for model input
    img_array /= 255.  # Normalize pixel values
    return img_array

# Classify the image
def classify_image(img_array):
    prediction = model.predict(img_array)
    predicted_class = prediction.argmax(axis=-1)[0]  # Get the index of the highest probability
    return int(predicted_class) 

# Route for handling image classification
@app.route('/', methods=['GET', 'POST'])
def classify():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        img_path = "uploads/" + file.filename
        file.save(img_path)

        # Preprocess input image
        input_img_array = preprocess_image(img_path)

        # Classify input image
        predicted_class = classify_image(input_img_array)

        return jsonify({'predicted_class': predicted_class})

    # For GET requests, render the HTML template
    return render_template('image process.html')

if __name__ == '__main__':
    app.run(debug=True)