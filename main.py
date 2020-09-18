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
embedder =FaceNet()
client=MongoClient()
mongodb=client.face_recognition

templates=Jinja2Templates(directory="templates")

class Input(BaseModel):
     base64str:str

if __name__ == "__main__":
    uvicorn.run(app,host="127.0.0.1",port=8000)

# # @app.on_event("startup")
# # async def create_db_client():


# # @app.on_event("shutdown")
# # async def shutdown_db_cliemt():
# #     client.close()
# def base64str_to_PILImage(base64str):
#     base64_img_bytes = base64str.encode('utf-8')
#     base64bytes = base64.b64decode(base64_img_bytes)
#     bytesObj = io.BytesIO(base64bytes)
#     img = Image.open(bytesObj)
#     return img

@app.get("/")
def home(request:Request):
    return templates.TemplateResponse("dashboard.html",{"request":request})

# @app.post("/files/")
# async def create_file(file: bytes = File(...),fileb: UploadFile = File(...), token: str = Form(...)):
#     return {
#         "file_size": len(file),
#         "token": token,
#         "fileb_content_type": fileb.content_type,
#     }

# @app.post("/boundingbox/")
# async def gen_boundingbox(file: UploadFile =File(...)):
#     readFile=await file.read()
#     return detect_faces(readFile)

# @app.post("/l2distance/")
# async def gen_l2distance(mainImage: UploadFile= File(...),refImage:UploadFile=File(...)):
#     FACENET_MODEL_PATH="model\\facenet_keras.h5"
#     FACENET_WEIGHTS_PATH='weights\\facenet_keras_weights.h5'
    
#     original_embedding=utils.get_embedding(embedder.model,originalFace)
#     test_embedding=utils.get_embedding(embedder.model,testFace)

#     dist=np.linalg.norm(test_embedding-original_embedding)

#     return {"l2Distance":float(dist)}


@app.post('/registerFace')
async def register_face(name: str,refImage:UploadFile=File(...)):
    face=utils.extract_face(await refImage.read())
    mongodb.embeddings.insert_one({"faceName":name,"embedding":Binary(pickle.dumps(utils.get_embedding(embedder.model, face),protocol=2),subtype=128)})

    return {"Success": "Face is registered successfully"}

# ,mainImage: UploadFile= File(...),
@app.post('/recognizeFace')
async def recognize_Face(mainImage: UploadFile= File(...)):
    # Extract Face
    face=utils.extract_face(await mainImage.read())

    # Generate Embedding
    embedding=utils.get_embedding(embedder.model,face)
    # get orginal embedding

    distance,identityIndex = faiss_utils.searchEmbedding(np.expand_dims(embedding,axis=0))

        
    if(distance < 0.8):
        prediction=mongodb.embeddings.find()[identityIndex]["faceName"]
    else:
        prediction="Unknown"


    print(prediction)   
    print(distance)   

    return {
        "Prediction": str(prediction),
        "L2Distance": float(distance)
    }
