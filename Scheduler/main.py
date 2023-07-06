from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import trainingQueue

# Start FastAPI
app = FastAPI()

# Defining APIs begin

@app.get("/")
async def root():
    return {"Server": "Scheduler"}

def fileify(f):
    return FileResponse(f)

@app.get("/api/v1/addToTrainingQueue/{user}/{ai}")
async def addToTrainingQueue(user: str, ai: str):
    trainingQueue.queueForRetraining(user, ai)
    #return {"labels": MultiLabelClassifier.class_names}

@app.get("/api/v1/getTrainingQueue/")
async def getTrainingQueue():
    return trainingQueue.getTrainingQueue()

@app.get("/api/v1/getFirstElementInQueue/")
async def getFirstElement():
    print("inside getFirstElement")
    queueElement = trainingQueue.getFirstElementInQueue()
    print("queueElement is ",queueElement)
    return queueElement


'''@app.post("/api/v1/upload-image/")
async def infer_label(url: str):
    return {"label": MultiLabelClassifier.inferUploadedImg(url)}

@app.post("/api/v1/add-label/")
async def add_label(label: str):
    MultiLabelClassifier.addLabel(label)
    return {"labels": MultiLabelClassifier.class_names}'''

# Defining APIs end