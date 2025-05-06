import streamlit as st
import os
import json
from PIL import Image, ImageDraw
import random

st.title("Moondream Search")

json_output_path = "results.json"

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
                if search_term.lower() in description.lower() or matched_classes:
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
