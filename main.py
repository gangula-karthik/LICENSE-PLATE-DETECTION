from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
from ultralytics import YOLO
import easyocr
from io import BytesIO
from PIL import Image
import numpy as np
import uvicorn


app = FastAPI()

model = YOLO('./model_weights/best.torchscript', task='detect')
reader = easyocr.Reader(['en'])

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
        detections = reader.readtext(plate_threshold)
        if detections:
            text, confidence = detections[0][1].upper().replace(' ', ''), detections[0][2]
            plates.append({'plate_number': text, 'confidence_score': confidence, 'bounding_box': {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}})

    return JSONResponse(content={"results": plates})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)