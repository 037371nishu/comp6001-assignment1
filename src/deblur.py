import cv2
import numpy as np


img = cv2.imread("data/raw/sample.png")

if img is None:
    print("❌ Image not found. Check path!")
    exit()


kernel = np.ones((5,5)) / 25
deblurred = cv2.filter2D(img, -1, kernel)


cv2.imwrite("data/deblurred/output.png", deblurred)

print("✅ Done")