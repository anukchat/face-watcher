import io
import pickle
import uuid

import cv2
import matplotlib.pyplot as plt
import numpy as np
import uvicorn
from bson.binary import Binary
from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
from keras.models import load_model
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN
from PIL import Image as PImage
from pydantic import BaseModel
from pymongo import MongoClient

import faiss_utils
import utils

app = FastAPI()
embedder = FaceNet()
client = MongoClient()
mongodb = client.face_recognition

templates = Jinja2Templates(directory="templates")


class Input(BaseModel):
    base64str: str


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.post('/registerFace')
async def register_face(name: str, refImage: UploadFile = File(...)):
    face = utils.extract_face(await refImage.read())
    mongodb.embeddings.insert_one({"faceName": name, "embedding": Binary(
        pickle.dumps(utils.get_embedding(embedder.model, face), protocol=2), subtype=128)})

    return {"Success": "Face is registered successfully"}

# ,mainImage: UploadFile= File(...),


@app.post('/recognizeFace')
async def recognize_Face(mainImage: UploadFile = File(...)):
    # Extract Face
    face = utils.extract_face(await mainImage.read())

    # Generate Embedding
    embedding = utils.get_embedding(embedder.model, face)
    # get orginal embedding

    distance, identityIndex = faiss_utils.searchEmbedding(
        np.expand_dims(embedding, axis=0))

    if(distance < 0.8):
        prediction = mongodb.embeddings.find()[identityIndex]["faceName"]
    else:
        prediction = "Unknown"

    print(prediction)
    print(distance)

    return {
        "Prediction": str(prediction),
        "L2Distance": float(distance)
    }
