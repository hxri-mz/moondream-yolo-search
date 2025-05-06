from ultralytics import YOLO

model = YOLO("yolo12n.pt")
results = model("/home/mz/moon/data/10.jpg")
import pdb; pdb.set_trace()