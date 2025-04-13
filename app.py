import os
import cv2
import numpy as np
import torch
import gradio as gr
from PIL import Image
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

# Model initialization


def initialize_model(model_path):
    print("Initializing model...")
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, None, model_name, use_flash_attn=True
    )
    return tokenizer, model, image_processor

# Process image and generate response from the model


def model_inference(tokenizer, model, image_processor, image, prompt):
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(image)
    image_size = pil_image.size

    # Process image for model input
    image_tensor = process_images(
        [pil_image], image_processor, model.config).half()

    # Prepare conversation prompt
    conv = conv_templates['v1'].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()

    # Tokenize input
    input_ids = tokenizer_image_token(
        full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    input_ids = input_ids.unsqueeze(0).to(model.device)

    # Generate response
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            do_sample=False,
            max_new_tokens=256,
            use_cache=True
        )

    output_text = tokenizer.decode(
        output_ids[0], skip_special_tokens=True).strip()
    return output_text


class Camera:
    def __init__(self):
        self.cap = None
        self.current_frame = None

    def start(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Error: Could not open camera.")
                return False
            return True
        return True

    def read_frame(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return self.current_frame
        return None

    def get_current_frame(self):
        return self.current_frame

    def stop(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


# Initialize model and camera
MODEL_PATH = "LLaRA/checkpoints/llava-1.5-7b-llara-D-inBC-Aux-B-VIMA-80k"
camera = Camera()
tokenizer, model, image_processor = None, None, None


def load_model():
    global tokenizer, model, image_processor
    tokenizer, model, image_processor = initialize_model(MODEL_PATH)
    return "Model loaded successfully!"


def start_camera():
    if camera.start():
        return "Camera started successfully!"
    return "Failed to start camera."


def process_frame():
    frame = camera.read_frame()
    if frame is not None:
        return frame
    return np.zeros((480, 640, 3), dtype=np.uint8)


def chat_with_llara(history, message):
    # Ensure camera and model are initialized
    if not camera.start():
        return history + [(message, "Error: Camera not available")]

    if tokenizer is None or model is None or image_processor is None:
        return history + [(message, "Error: Model not initialized. Please load the model first.")]

    # Get current frame
    frame = camera.get_current_frame()
    if frame is None:
        return history + [(message, "Error: No camera frame available")]

    # Process with model
    try:
        response = model_inference(
            tokenizer, model, image_processor, frame, message)
        return history + [(message, response)]
    except Exception as e:
        return history + [(message, f"Error processing request: {str(e)}")]


def cleanup():
    camera.stop()
    return "Camera stopped."


with gr.Blocks() as app:
    gr.Markdown("# LLaRA Robotics Vision-Language Assistant")

    with gr.Row():
        with gr.Column(scale=2):
            video_output = gr.Image(label="Live Camera Feed")

            with gr.Row():
                load_model_btn = gr.Button("1. Load Model")
                start_camera_btn = gr.Button("2. Start Camera")
                stop_camera_btn = gr.Button("Stop Camera")

        with gr.Column(scale=1):
            chatbot = gr.Chatbot(height=480)
            msg = gr.Textbox(
                placeholder="Ask something about the camera view...", show_label=False)
            clear = gr.Button("Clear")

    # Set up event handlers
    load_model_btn.click(load_model, inputs=[], outputs=gr.Textbox())
    start_camera_btn.click(start_camera, inputs=[], outputs=gr.Textbox())
    stop_camera_btn.click(cleanup, inputs=[], outputs=gr.Textbox())

    # Update frame every 0.1 second
    video_output.update(process_frame, every=0.1)

    # Chat functionality
    msg.submit(chat_with_llara, [chatbot, msg], [chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

app.queue()
app.launch(share=True)
