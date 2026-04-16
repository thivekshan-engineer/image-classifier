"""
Image Classifier using TensorFlow & MobileNetV2
Author: Thivekshan Rajakumar
Description: A deep learning image classifier that uses a pre-trained 
MobileNetV2 model to classify images into 1000 categories.
"""
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import sys
import os
import time

# ─────────────────────────────────────────
# Colour output for terminal
# ─────────────────────────────────────────
GREEN  = "\033[92m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def print_banner():
    print(f"""
{CYAN}{BOLD}
╔══════════════════════════════════════════════╗
║        AI IMAGE CLASSIFIER                  ║
║        Powered by TensorFlow & MobileNetV2  ║
║        by Thivekshan Rajakumar              ║
╚══════════════════════════════════════════════╝
{RESET}""")

def load_model():
    """Load the pre-trained MobileNetV2 model."""
    print(f"{YELLOW}⏳ Loading pre-trained MobileNetV2 model...{RESET}")
    start = time.time()
    model = MobileNetV2(weights='imagenet')
    elapsed = time.time() - start
    print(f"{GREEN}✅ Model loaded in {elapsed:.2f} seconds{RESET}\n")
    return model

def preprocess_image(img_path):
    """Load and preprocess an image for classification."""
    if not os.path.exists(img_path):
        print(f"{RED}❌ Error: File '{img_path}' not found.{RESET}")
        sys.exit(1)

    print(f"{CYAN}📷 Loading image: {img_path}{RESET}")
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def classify_image(model, img_array, top_k=5):
    """Run prediction and return top K results."""
    print(f"{YELLOW}🧠 Running AI classification...{RESET}\n")
    predictions = model.predict(img_array, verbose=0)
    results = decode_predictions(predictions, top=top_k)[0]
    return results

def display_results(results, img_path):
    """Display classification results in a clean format."""
    print(f"{BOLD}{CYAN}{'─'*50}{RESET}")
    print(f"{BOLD}🔍 Classification Results for: {os.path.basename(img_path)}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*50}{RESET}\n")

    for rank, (class_id, label, confidence) in enumerate(results, 1):
        bar_length = int(confidence * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        confidence_pct = confidence * 100

        if rank == 1:
            color = GREEN
            medal = "🥇"
        elif rank == 2:
            color = CYAN
            medal = "🥈"
        elif rank == 3:
            color = YELLOW
            medal = "🥉"
        else:
            color = RESET
            medal = f" {rank}."

        print(f"{color}{medal} {label.replace('_', ' ').title():<30} {bar} {confidence_pct:.2f}%{RESET}")

    print(f"\n{BOLD}{GREEN}✅ Top prediction: {results[0][1].replace('_', ' ').title()} "
          f"({results[0][2]*100:.2f}% confidence){RESET}")
    print(f"{CYAN}{'─'*50}{RESET}\n")

def get_image_info(img_path):
    """Display basic image information."""
    img = Image.open(img_path)
    width, height = img.size
    mode = img.mode
    size_kb = os.path.getsize(img_path) / 1024
    print(f"{CYAN}📊 Image Info:{RESET}")
    print(f"   • Dimensions : {width} x {height} pixels")
    print(f"   • Mode       : {mode}")
    print(f"   • File size  : {size_kb:.1f} KB\n")

def main():
    print_banner()

    # Get image path from command line or prompt user
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = input(f"{YELLOW}📁 Enter the path to your image: {RESET}").strip()

    # Display image info
    get_image_info(img_path)

    # Load model
    model = load_model()

    # Preprocess image
    img_array = preprocess_image(img_path)

    # Classify
    results = classify_image(model, img_array, top_k=5)

    # Display results
    display_results(results, img_path)

if __name__ == "__main__":
    main()