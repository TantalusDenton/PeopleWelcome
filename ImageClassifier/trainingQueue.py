from queue import Queue

# need a queue to put train requests
trainingQueue = Queue()

def queueForRetraining(user, ai):
  data = [user, ai]
  trainingQueue.put(data)
  print(trainingQueue)
  # mlClassifier = newMultiLabelClassifier()
  # mlClassifier.retrain()