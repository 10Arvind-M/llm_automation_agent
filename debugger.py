'''import os
from dotenv import load_dotenv

load_dotenv()
print(os.getenv("AIPROXY_TOKEN"))  # Should print your actual key
import cv2
print("OpenCV version:", cv2.__version__)'''
import pytesseract
print("Tesseract Version:", pytesseract.get_tesseract_version())
import cv2
print(cv2.getversion())