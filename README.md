# LLaRA Robotics Vision-Language Assistant

A Gradio web interface that connects a live camera feed with the LLaRA (Language-LLM-as-Robot-Assistant) vision-language model. This application allows you to ask questions about objects in the camera view and get intelligent responses from the model.

## Features

- Real-time camera feed display
- Chat interface to interact with the LLaRA model
- Send live camera frames to the model for visual reasoning
- Responsive UI with side-by-side video and chat

## Prerequisites

1. **Python Environment**: Python 3.10+ recommended
2. **Dependencies**: Listed in requirements.txt
3. **LLaRA Model**: Download the pretrained model from Hugging Face

## Setup Instructions

### 1. Clone this repository and the LLaRA repository

```bash
git clone https://github.com/yourusername/robotics-gradio-website.git
cd robotics-gradio-website
git clone https://github.com/LostXine/LLaRA.git
```

### 2. Set up Python environment

```bash
conda create -n llara python=3.10 -y
conda activate llara
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install cuda=12.1 cuda-compiler=12.1 cuda-nvcc=12.1 cuda-version=12.1 -c nvidia
```

### 3. Install LLaVA (required by LLaRA)

```bash
cd LLaRA/train-llava && pip install -e ".[train]"
pip install flash-attn==2.7.3 --no-build-isolation
cd ../..
```

### 4. Install other requirements

```bash
pip install -r requirements.txt
```

### 5. Download the pretrained model

Download the LLaRA model from Hugging Face and place it in the `LLaRA/checkpoints/` directory:

- [llava-1.5-7b-llara-D-inBC-Aux-B-VIMA-80k](https://huggingface.co/variante/llava-1.5-7b-llara-D-inBC-Aux-B-VIMA-80k)

## Running the Application

```bash
python app.py
```

This will start the Gradio interface. Follow these steps:

1. Click "1. Load Model" to initialize the LLaRA model
2. Click "2. Start Camera" to begin the camera feed
3. Use the chat interface to ask questions about objects in the camera view

## Usage Examples

- "What objects do you see in the camera view?"
- "Can you identify the red object?"
- "How would you pick up the cup on the table?"
- "What's the best way to grasp the object on the left?"

## Note

The LLaRA model is specifically designed for robotics applications and performs best with questions about object manipulation, spatial reasoning, and robot action planning.
