import streamlit as st
import os
import json
from PIL import Image
import numpy as np
from transformers import AutoModelForCausalLM
from ultralytics import YOLO
from PIL import Image, ImageDraw
import random
import torch

torch.classes.__path__ = []

@st.cache_resource
def load_model():
    return AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        revision="2025-01-09",
        trust_remote_code=True,
        device_map={"": "cuda"}
    )

model = load_model()

def random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

class_colors = {}

def draw_bounding_boxes(image_path, detections):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    for obj in detections:
        box = obj['box']
        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
        
        class_name = obj['name']
        
        if class_name not in class_colors:
            class_colors[class_name] = random_color()
        
        color = class_colors[class_name]        
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1, y1), class_name, fill=color)
    
    return image

def format_matched_classes(detections):
    class_count = {}
    for obj in detections:
        class_name = obj['name']
        class_count[class_name] = class_count.get(class_name, 0) + 1
    
    class_summary = [f"{count} {class_name}" for class_name, count in class_count.items()]
    return ', '.join(class_summary)

yolo_model = YOLO("yolo12n.pt")
st.title("Moondream Search")

json_output_path = "results.json"
output_folder = "output"

options = st.radio(
    "Select one to continue",
    ["Process data", "Search"],
    captions=['', 'Make sure you process data first if not done.',]
)

if options == "Process data":
    folder_path = st.text_input("Enter image folder path", value="data/")
    if st.button("Process Images"):
        if not os.path.exists(folder_path):
            st.error("Folder path does not exist.")
        else:
            os.makedirs(output_folder, exist_ok=True)
            results = {}

            image_files = [f for f in os.listdir(folder_path)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            total_images = len(image_files)

            progress_bar = st.progress(0)

            for idx, fname in enumerate(image_files):
                print(f'processing {idx}')
                image_path = os.path.join(folder_path, fname)
                image = Image.open(image_path).convert("RGB")
                encoded_image = model.encode_image(image)

                try:
                    detected_objects = json.loads(yolo_model(image)[0].to_json())

                    description = model.query(encoded_image,
                        "Describe the scene with objects and their colors, number of lanes, "
                        "whether it's marked or not, weather (sunny, rainy, snow, cloudy), "
                        "and time of day (morning, night, dawn, dusk). Mention pedestrians if present.")

                    results[fname] = {
                        "detections": detected_objects,
                        "description": description
                    }

                except Exception as e:
                    st.warning(f"Error processing {fname}: {e}")
                    continue

            with open(json_output_path, "w") as f:
                json.dump(results, f, indent=4)

            st.success(f"Processing complete. Output saved to {json_output_path}")
else:
    search_term = st.text_input("Enter search term", "")
    if st.button("Search"):
        if search_term:
            if os.path.exists(json_output_path):
                with open(json_output_path, "r") as f:
                    results = json.load(f)

                matches = []
                for filename, data in results.items():
                    description = data.get('description', "")['answer']
                    detections = data.get('detections', [])                
                    matched_classes = [
                        obj['name'] for obj in detections 
                        if search_term.lower() in str(obj['name']).lower()
                    ]
                    if search_term.lower() in description.lower().split() or matched_classes:
                        matches.append({
                            "filename": filename,
                            "description": description,
                            "detections": detections,
                            "matched_classes": matched_classes
                        })
                
                if matches:
                    st.subheader("Matches Found:")
                    for match in matches:
                        formatted_classes = format_matched_classes(match['detections'])
                        
                        st.write(f"File: {match['filename']}")
                        st.write(f"Description: {match['description']}")
                        st.write(f"Matched Classes: {formatted_classes}")
                        
                        image_path = os.path.join("data", match['filename'])
                        if os.path.exists(image_path):
                            image_with_bboxes = draw_bounding_boxes(image_path, match['detections'])
                            
                            st.image(image_with_bboxes, caption=f"Image with detections: {match['filename']}")
                        else:
                            st.warning(f"Image file {match['filename']} not found.")
                else:
                    st.write("No matches found.")
            else:
                st.error(f"No results found. Please process images first.")
