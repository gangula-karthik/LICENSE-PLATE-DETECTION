from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
from ultralytics import YOLO
from io import BytesIO
from PIL import Image
import numpy as np
import uvicorn
from paddleocr import PaddleOCR


app = FastAPI()

model = YOLO('./model_weights/best.torchscript', task='detect')
reader = PaddleOCR(lang='en') # need to run only once to load model into memory

@app.post("/detect-license-plate/")
async def detect_license_plate(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert('RGB')  
    
    results = model(np.array(image))
    plates = []

    for result in results:
        bbox = result.boxes.xyxy[0]
        xmin, ymin, xmax, ymax = map(int, bbox)
        
        # crop image to license plate only
        plate_image = np.array(image.crop((xmin, ymin, xmax, ymax)).convert('L'))
        _, plate_threshold = cv2.threshold(plate_image, 64, 255, cv2.THRESH_BINARY_INV)
        
        # read the license plate
        result = reader.ocr(plate_threshold, det=False, cls=False)
        # Concatenate the plate strings
        plate_numbers = " ".join(plate for res in result for plate, _ in res)
        print(plates)
        # Collect and average the scores
        scores = [score for res in result for _, score in res]
        average_score = sum(scores) / len(scores) if scores else 0
        plates.append({'plate_number': plate_numbers, 'confidence_score': average_score, 'bounding_box': {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}})

    return JSONResponse(content={"results": plates})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)