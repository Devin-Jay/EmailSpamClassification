import math
import random
import numpy as np
from sklearn.metrics import pairwise_distances

class KNearestNeighbor:
    def __init__(self, attributes):
        self.labels = attributes
        self.numOfXVals = len(self.labels) - 1
    
    def evalEmailEucl(self, evalEmail, trainingSet):
        # index 0 is euclidian distance; index 1 is email's class
        euclidianDistance = (100000,0)
        for email in trainingSet:
            distance = 0
            for attribute in range(self.numOfXVals):
                distance += (float(evalEmail[attribute]) - float(email[attribute])) ** 2
                if (math.sqrt(distance) > euclidianDistance[0]):
                    break

            if (math.sqrt(distance) < euclidianDistance[0]):
                euclidianDistance = (distance, int(email[-1]))
        
        return euclidianDistance[1]  

    def classifySet(self, evalSet, trainingSet):
        result = []
        for email in evalSet:
            pred = self.evalEmailEucl(email, trainingSet)
            if (pred > 0.5):
                result.append(1)
            else:
                result.append(0)

        return result