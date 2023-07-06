from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from models import Character, Image
from uuid import UUID, uuid4
import MultiLabelClassifier
#import trainingQueue

# Start FastAPI
app = FastAPI()

# Defining APIs begin

@app.get("/")
async def root():
    return {"People": "Welcome"}

def fileify(f):
    return FileResponse(f)

@app.get("/api/v1/retrain/{user}/{ai}")
async def retrain(user: str, ai: str):
    MultiLabelClassifier.reTrain(user, ai)
    #return {"labels": MultiLabelClassifier.class_names}

'''@app.post("/api/v1/upload-image/")
async def infer_label(url: str):
    return {"label": MultiLabelClassifier.inferUploadedImg(url)}

@app.post("/api/v1/add-label/")
async def add_label(label: str):
    MultiLabelClassifier.addLabel(label)
    return {"labels": MultiLabelClassifier.class_names}'''

# Defining APIs end