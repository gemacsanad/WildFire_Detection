import gradio as gr
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# Preprocessing function (adjusted to work on an image array)
def preprocess_image(image, image_size=(128, 128)):
    if image is None:
        return None
    # Convert PIL image to OpenCV format (numpy array)
    image = np.array(image.convert('RGB'))
    image = cv2.resize(image, image_size)
    image = image / 255.0
    return np.expand_dims(image.astype(np.float32), axis=0)  # Add batch dim

# Prediction function for Gradio interface
def predict(image, model_name):
    if image is None:
        return "No image uploaded."
    try:
        model = tf.keras.models.load_model(model_name)
    except Exception as e:
        return f"Error loading model: {e}"

    X = preprocess_image(image)
    if X is None:
        return "Error preprocessing image."

    prediction = model.predict(X)
    predicted_class = np.argmax(prediction[0])

    if predicted_class == 0:
        return "No Wildfire Detected"
    else:
        return "Wildfire Detected"


# List of model files (make sure these files are in the same directory or give full paths)
models = ["small_model.h5", "medium_model.h5", "model.h5"]

# Gradio UI layout
with gr.Blocks() as demo:
    gr.Markdown("# Model Predictor")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Image")
        model_dropdown = gr.Dropdown(choices=models, label="Select Model")

    predict_button = gr.Button("Predict")
    output_text = gr.Textbox(label="Prediction Result", interactive=False)

    predict_button.click(
        fn=predict,
        inputs=[image_input, model_dropdown],
        outputs=output_text
    )

if __name__ == "__main__":
    demo.launch()
