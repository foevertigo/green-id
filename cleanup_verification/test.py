from ultralytics import YOLO
from PIL import Image

model = YOLO("taco_best.pt")
img = Image.open("after.png") 

results = model(img)
results[0].show()  
