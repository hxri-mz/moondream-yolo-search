# import moondream as md
# from PIL import Image
# import numpy as np
# import cv2

# model = md.vl(model="/home/hxri/moon/ckpt/moondream-2b-int8.mf.gz")
# image = Image.open("/home/hxri/moon/data/14.jpg")
# encoded_image = model.encode_image(image)

# def draw_bbox(image: np.ndarray, bbox: list, color=(0, 255, 0), thickness=2):
#     img_h, img_w = image.shape[:2]
#     for box in bbox:
#         x_min = int(box['x_min'] * img_w)
#         y_min = int(box['y_min'] * img_h)
#         x_max = int(box['x_max'] * img_w)
#         y_max = int(box['y_max'] * img_h)
#         cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
#     return image

# objects = model.detect(encoded_image, "cow")["objects"]
# proj = draw_bbox(image=np.array(image), bbox=objects)
# cv2.imwrite('out.png', cv2.cvtColor(proj, cv2.COLOR_BGR2RGB))
# print(f"Found {len(objects)} cow(s)")
# desc = model.query(encoded_image, "Describe the scene with objects and their colors and number of lanes and if its marked or not marked and weather (sunny, rainy, snow, cloudy) and time of day (morning, night, dawn, dusk). Also only mention pedestrians if they are present.")
# print(f"Description: {desc}")


import streamlit as st
import os
import json
from PIL import Image
import numpy as np
import cv2
from transformers import AutoModelForCausalLM

@st.cache_resource
def load_model():
    # return md.vl(model="/home/mz/moon/ckpt/moondream-2b-int8.mf.gz")
    return AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        revision="2025-01-09",
        trust_remote_code=True,
        device_map={"": "cuda"}
    )

model = load_model()

def draw_bbox(image: np.ndarray, bbox: list, color=(0, 255, 0), thickness=2):
    img_h, img_w = image.shape[:2]
    for box in bbox:
        x_min = int(box['x_min'] * img_w)
        y_min = int(box['y_min'] * img_h)
        x_max = int(box['x_max'] * img_w)
        y_max = int(box['y_max'] * img_h)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
    return image

st.title("Moondream search")

folder_path = st.text_input("Enter image folder path", value="data/")
query_object = st.text_input("Object to detect", value="cow")
json_output_path = "results.json"
output_folder = "output"

# if st.button("Search"):
#     st.success(f"Searching")

if st.button("Process Images"):
    if not os.path.exists(folder_path):
        st.error("Folder path does not exist.")
    else:
        os.makedirs(output_folder, exist_ok=True)
        results = {}

        image_files = [f for f in os.listdir(folder_path)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for fname in image_files:
            image_path = os.path.join(folder_path, fname)
            image = Image.open(image_path).convert("RGB")
            encoded_image = model.encode_image(image)

            try:
                objects = model.detect(encoded_image, query_object)["objects"]
                description = model.query(encoded_image,
                    "Describe the scene with objects and their colors and number of lanes and "
                    "if it's marked or not marked and weather (sunny, rainy, snow, cloudy) and "
                    "time of day (morning, night, dawn, dusk). Also only mention pedestrians if they are present.")

                # results[fname] = {
                #     "detections": objects,
                #     "description": description
                # }

                results[fname] = {
                    "detections": objects,
                    "description": description
                }

                np_image = np.array(image)
                proj = draw_bbox(np_image.copy(), bbox=objects)
                save_path = os.path.join(output_folder, f"bbox_{fname}")
                cv2.imwrite(save_path, cv2.cvtColor(proj, cv2.COLOR_RGB2BGR))

                st.image(proj, caption=f"{fname} — {len(objects)} object(s) found")

            except Exception as e:
                st.warning(f"Error processing {fname}: {e}")
                continue

        with open(json_output_path, "w") as f:
            json.dump(results, f, indent=4)

        st.success(f"Processing complete. Output saved to {json_output_path}")