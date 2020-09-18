from pymongo import MongoClient
import pickle
import faiss
import numpy as np

client=MongoClient()
mongodb=client.face_recognition
d=512
k=1

def fetchAllEmbeddings():
    
    allEmbedding=mongodb.embeddings.find()
    embeddings= [pickle.loads(record["embedding"]) for record in allEmbedding]

    index = faiss.IndexFlatL2(d)   # build the index
    print(index.is_trained)

    embeddingArr=np.array(embeddings)
    # embeddingArr=np.expand_dims(embeddingArr,axis=0)
    print(embeddingArr.shape)

    index.add(embeddingArr)                  # add vectors to the index

    return index

def searchEmbedding(embedding):
    
    index=fetchAllEmbeddings()
    D, I = index.search(embedding, k) # sanity check
    print(I)
    print(D)

    return np.asscalar(D),np.asscalar(I)        

