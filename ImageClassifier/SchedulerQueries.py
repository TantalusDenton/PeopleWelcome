import requests
import json

def getFirstElementInQueue():
    training_queue_api_url = "http://localhost:3003/api/v1/getFirstElementInQueue/"
    res = requests.get(training_queue_api_url)
    data = json.loads(res.text)
    return data
