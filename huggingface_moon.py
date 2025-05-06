from transformers import AutoModelForCausalLM
from PIL import Image
import time

model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-01-09",
    trust_remote_code=True,
    device_map={"": "cuda"}
)
start = time.time()
image = Image.open('/home/mz/moon/data/1.jpg')
enc_image = model.encode_image(image)
print(model.query(enc_image, "Describe this image."))
end = time.time()

print(f'Inference time = {end-start:.2f} seconds')
