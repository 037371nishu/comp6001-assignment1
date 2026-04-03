from ultralytics import YOLO


model = YOLO("yolov8n.pt")


print("Running detection on BLURRED image...")
model("data/raw/sample.png", save=True)


print("Running detection on DEBLURRED image...")
model("data/deblurred/output.png", save=True)

print("✅ Detection complete")