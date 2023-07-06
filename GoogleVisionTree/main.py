from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ObjectDetection import getSuggestionsByAi, retrainTree
from pydantic import BaseModel


class Body(BaseModel):
    id_list: list

# Start FastAPI
app = FastAPI()

origins = [
    "http://localhost:3000",
    "https://applogic.wwwelco.me:5000",
    "https://people.wwwelco.me",
    "https://objectdetector.wwwelco.me:3002",
]



app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/account/{user}/{ai}/{id}/tagsuggestions')
async def getSuggestions(user: str, ai: str, id: str):
    ai = ai.replace(" ", "%20")
    return {'suggestions': getSuggestionsByAi(user, ai, [id])}

@app.post('/account/{user}/{ai}/retraintree')
async def trainTree(user: str, ai: str):
    ai = ai.replace(" ", "%20")
    retrainTree(user, ai)

