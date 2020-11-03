# face-watcher

This repository contains end to end implementation of Face Recognition using FaceNet and MTCNN

### Requirements

1. Need Mongo DB Instance (Used here to store 128 D face embeddings)

### Usage

1. Installation: pip install -r requirements.txt
2. uvicorn main:app --reload

### Description

- Facial Verification system is a system where by just inputing a the image of the person its identity
  can be extracted ,
- System captures image of a person, recognizes and extract faces and then
  generate a 128 D vector embedding (feature vector),
- A similarity score is than calculated with the reference image, the score is then compared with a threshold and then the person is identified.
- Here for calculation of Similarity score FAISS (Facebook AI Similarity Search) library is used
- Faiss is built around an index type that stores a set of vectors, and provides a function to search in them with L2 and/or dot product vector comparison.

### Architecture
