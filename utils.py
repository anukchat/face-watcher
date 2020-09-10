# get embedding function will return the 128 embedding
from PIL import Image
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
import numpy as np
import io
import uuid

def get_embedding(model,face_pixels):

    # scale pixel wvalues
    face_pixels=face_pixels.astype('float32')
    mean,std=face_pixels.mean(),face_pixels.std()
    face_pixels=(face_pixels-mean)/std

    samples=np.expand_dims(face_pixels,axis=0)

    yhat=model.predict(samples)

    return yhat[0]

# will return the face coordinates
def extract_face(filename, required_size=(160, 160)):
    image=Image.open(io.BytesIO(filename))

    image=image.convert('RGB')

    pixels=np.asarray(image)

    detector=MTCNN()

    results=detector.detect_faces(pixels)

    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array

def detect_faces(image_path):

    image = PImage.open(io.BytesIO(image_path))
    image = image.convert('RGB')
    pixels = np.asarray(image)

    detector = MTCNN()
    # detect faces in the image

    results = detector.detect_faces(pixels)

    # extract the bounding box from the faces
    detected_faces = list()
    for result in results:

        # only detect faces with a confidence of 90% and above
        if result['confidence'] > 0.90:
            detected_faces.append({
                "face_id":uuid.uuid4(),
                "confidence": result['confidence'],
                "bounding_box": result['box'],
                "keypoints":result['keypoints']
            })

    return detected_faces

