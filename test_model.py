from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the trained model
model = load_model('model1.h5')

# Function to preprocess an image for prediction
def preprocess_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file '{image_path}' not found.")
    # Load the image in grayscale mode
    img = load_img(image_path, color_mode="grayscale", target_size=(28, 28))
    # Convert the image to an array and normalize it
    img_array = img_to_array(img) / 255.0
    # Reshape the array to match the input shape of the model
    img_array = img_array.reshape(1, 28, 28, 1)

    # Visualize the preprocessed image
    plt.imshow(img_array.reshape(28, 28), cmap='gray')
    plt.title("Preprocessed Input Image")
    plt.show()

    return img_array

# Function to predict the digit and show accuracy
def predict_digit(image_path):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)
    confidence_score = prediction[0][predicted_digit] * 100  # Confidence score for the predicted digit
    print(f"Predicted Digit: {predicted_digit}")
    print(f"Prediction Confidence Score: {confidence_score:.2f}%")
    print("All Prediction Scores:", prediction)

# Main loop for user interaction
if __name__ == "__main__":
    print("Improved model is ready. Enter the path to an image file (or type 'exit' to quit):")
    while True:
        image_path = input("Image Path: ").strip('"').strip("'")
        if image_path.lower() == 'exit':
            print("Exiting...")
            break
        try:
            predict_digit(image_path)
        except Exception as e:
            print(f"Error: {e}. Please try again with a valid image file.")
