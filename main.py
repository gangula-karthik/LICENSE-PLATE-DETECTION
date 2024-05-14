from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
from ultralytics import YOLO
from io import BytesIO
from PIL import Image
import numpy as np
import uvicorn
from typing import List, Dict
from paddleocr import PaddleOCR


app = FastAPI()

# Load models
model = YOLO('./model_weights/best.torchscript', task='detect')
reader = PaddleOCR(lang='en')

@app.post("/detect-license-plate/")
async def detect_license_plate(file: UploadFile = File(...)) -> JSONResponse:
    image = Image.open(BytesIO(await file.read())).convert('RGB')
    detections = model(np.array(image))
    plates = [process_plate(image, det) for det in detections]

    return JSONResponse(content={"results": plates})

def process_plate(image: Image.Image, detection: Dict) -> Dict:
    """Extracts, processes, and reads license plate from a detected bounding box."""
    bbox = map(int, detection['boxes'].xyxy[0])
    xmin, ymin, xmax, ymax = bbox
    plate_image = extract_plate(image, xmin, ymin, xmax, ymax)
    plate_text, confidence = read_plate(plate_image)

    return {
        'plate_number': plate_text,
        'confidence_score': confidence,
        'bounding_box': {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
    }

def extract_plate(image: Image.Image, xmin: int, ymin: int, xmax: int, ymax: int) -> np.ndarray:
    """Crops and preprocesses the plate image for OCR."""
    plate_image = np.array(image.crop((xmin, ymin, xmax, ymax)).convert('L'))
    _, plate_thresh = cv2.threshold(plate_image, 64, 255, cv2.THRESH_BINARY_INV)
    return plate_thresh

def read_plate(plate_image: np.ndarray) -> (str, float):
    """Performs OCR on the license plate and returns the text with average confidence."""
    ocr_results = reader.ocr(plate_image, det=False, cls=False)
    if not ocr_results:
        return "", 0.0

    plates = " ".join(plate for res in ocr_results for plate, _ in res)
    scores = [score for res in ocr_results for _, score in res]
    average_score = sum(scores) / len(scores)
    return plates, average_score

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
