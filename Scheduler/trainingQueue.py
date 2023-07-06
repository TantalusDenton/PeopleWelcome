from queue import Queue

# need a queue to put train requests
trainingQueue = Queue()

def queueForRetraining(user, ai):
  data = [user, ai]
  trainingQueue.put(data)
  print(trainingQueue)
  # mlClassifier = newMultiLabelClassifier()
  # mlClassifier.retrain()

def getTrainingQueue():
  print ('trainingQueue is')
  print(trainingQueue)
  return trainingQueue

def getFirstElementInQueue():
  if trainingQueue.empty():
    return ['status:','empty']
  else:
    element = trainingQueue.get()
    print ('trainingQueue element is')
    print(element)
    return element

# make a function to traverse the queue, if the tuple already exists, don't add