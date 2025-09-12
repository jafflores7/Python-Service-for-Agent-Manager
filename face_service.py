import os
import base64
import numpy as np
import face_recognition
from fastapi import FastAPI
from pydantic import BaseModel
from collections import defaultdict
import uvicorn
import sqlite3

# LSH Database Class
class LSH_DB:
    def __init__(self, dim, num_planes=10, db_path='lsh.db', data_path='vectors.npy'):
        self.num_planes = num_planes
        self.dim = dim
        self.datapath = os.path.abspath(data_path)


        try:
            self.planes = np.load('planes.npy')
        except FileNotFoundError:
            self.planes = np.random.randn(num_planes, dim)
            np.save('planes.npy', self.planes)
            
        # Setup SQLite Database
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS buckets 
                               (hash_key TEXT, vector_id INTEGER, label TEXT, active BOOL)''')
        self.conn.commit()
        
        # Setup the vector data file
        try:
            self.vectors = np.load(self.datapath, mmap_mode='r+') # Memory-mapped!
        except FileNotFoundError:
            # Create an empty file with the right dimensions
            self.vectors = np.zeros((0, dim)) # Start with an empty array
            np.save(self.datapath, self.vectors)
            self.vectors = np.load(self.datapath, mmap_mode='r+')
        self.next_id = len(self.vectors)

    def _hash(self, v):
        return str(tuple((v @ self.planes.T) > 0))  

    def add(self, v, label):
        print("Saving vectors to:", self.datapath)
        v = np.array(v).flatten()
        h = self._hash(v)
        
        # Append vector to the data file
        self.vectors = np.array(self.vectors)
        new_vectors = np.vstack([self.vectors, v])
        
        with open(self.datapath, "wb") as f:
            np.save(f, new_vectors)

        self.vectors = np.load(self.datapath, mmap_mode='r+') 
        
        # Add to database
        self.cursor.execute("INSERT INTO buckets VALUES (?, ?, ?, ?)", (h, self.next_id, label, True))
        self.conn.commit()
        self.next_id += 1

    def query(self, q, top_k=3):
        q = np.array(q).flatten()
        h = self._hash(q)
        
        # 1. Get candidate IDs and labels from the DB
        self.cursor.execute("SELECT vector_id, label FROM buckets WHERE hash_key=? AND active=TRUE", (h,))
        candidates = self.cursor.fetchall()
        
        if not candidates:
            return [], []
            
        # 2. Get the actual vectors from the data file using their IDs
        vector_ids = [id for id, label in candidates]
        candidate_vectors = self.vectors[vector_ids] 
        
        # 3. Calculate distances and return top_k
        dists = np.linalg.norm(candidate_vectors - q, axis=1)
        top_k_indices = np.argsort(dists)[:top_k]
        
        encodings = candidate_vectors[top_k_indices]
        labels = [candidates[i][1] for i in top_k_indices]
        return encodings, labels

    def deactivateAccess(self, q, label): 
        q = np.array(q).flatten()
        h = self._hash(q)
        self.cursor.execute("UPDATE buckets SET active = FALSE WHERE hash_key=? AND label=?", (h, label))
        return 

# -------------------------------
# Database Initialization
# -------------------------------
face_db = LSH_DB(128, 5)

#--------------------------------
#Functions to process data 
# -------------------------------

def create_new_reference(path, name):
    img = face_recognition.load_image_file(path)
    face_db.add(face_recognition.face_encodings(img)[0], name)

def delete_reference(path, name): 
    img = face_recognition.load_image_file(path)
    face_db.deactivateAccess(face_recognition.face_encodings(img)[0], name)
    return

#--------------------------------
#Example of how to add data
# -------------------------------
"""
create_new_reference("images/carlos.jpg", "Carlos")
create_new_reference("images/felipe.jpg", "Felipe")
create_new_reference("images/payday.jpg", "Payday")
create_new_reference("images/kepler.jpg", "Kepler")
"""

# FastAPI Service
app = FastAPI(title="Face Recognition Service", version="1.0.0")

class FaceInput(BaseModel):
    face_image: str  # base64 encoded image

class FaceInputWLabel(BaseModel):
    face_image: str #Another b64 img
    label: str

@app.get("/health")
def health():
    return {"status": "ok", "name": "face-recognition-agent", "version": "1.0.0"}


@app.post("/execute")
def execute(input: FaceInput):
    try:
        # Decode base64 image
        face_bytes = base64.b64decode(input.face_image)
        temp_path = "temp_input.jpg"
        with open(temp_path, "wb") as f:
            f.write(face_bytes)

        # Load face encoding
        image = face_recognition.load_image_file(temp_path)
        encodings = face_recognition.face_encodings(image)
        if not encodings:
            return {"status": "error", "message": "No face detected in input image"}

        query_encoding = encodings[0]
        candidates, labels = face_db.query(query_encoding)

        if len(candidates) == 0:
            return {"status": "success", "data": {"match": False, "label": None}}

        # Boolean matches
        matches = face_recognition.compare_faces(candidates, query_encoding, tolerance=0.5)

        # Distances (lower = closer match)
        distances = face_recognition.face_distance(candidates, query_encoding)

        # Always compute percentage (not tied to tolerance)
        def face_confidence(distance):
            similarity = (1 - distance) * 100
            return round(max(0.0, min(100.0, similarity)), 2)

        confidences = [face_confidence(d) for d in distances]

        results = [
            {"label": label, "match": bool(match), "match_rate": confidence}
            for label, match, confidence in zip(labels, matches, confidences)
        ]

        return {
            "status": "success",
            "data": {"matches": results}
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/add")
def add(data: FaceInputWLabel): 
    try: 
        face = base64.b64decode(data.face_image)
        temp_path = "face_to_add.jpg"
        with open(temp_path, "wb") as f: 
            f.write(face)
        v = face_recognition.face_encodings(face_recognition.load_image_file(temp_path))
        if len(v) == 0:
            return {"status": "error", "message": "No face detected in input image"}
        face_db.add(v, data.label)
        return { "status": "success", "data": {"Person added" : data.label, "v" : v[0].tolist() }}
    except Exception as e:
        return {"status": "error", "message": str(e)}
   
@app.post("/deleteAccess")
def add(data: FaceInputWLabel): 
    try: 
        face = base64.b64decode(data.face_image)
        temp_path = "face_to_deny.jpg"
        with open(temp_path, "wb") as f: 
            f.write(face)
        v = face_recognition.face_encodings(face_recognition.load_image_file(temp_path))
        if len(v) == 0:
            return {"status": "error", "message": "No face detected in input image"}
        face_db.deactivateAccess(v, data.label)
        return { "status": "success", "data": {"Person denied" : data.label, "v" : v[0].tolist() }}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    return 
# -------------------------------
# Run Server
# -------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", 3000)))