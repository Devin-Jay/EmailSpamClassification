import math
import random
from sklearn.metrics import pairwise_distances
import numpy as np

class KNearestNeighbor:
    def __init__(self, attributes):
        self.labels = attributes
        self.numOfXVals = len(self.labels) - 1 

    def classifySet(self, evalSet, trainingSet):
        # get trainingSet without target column
        trainingEmails = [row[:-1] for row in trainingSet]

        # get evalSet without target column
        if (len(evalSet[0]) == len(trainingSet[0])):
            evalEmails = [row[:-1] for row in evalSet]

        # calculate distance between each email in evalSet and trainingEmails
        distances = pairwise_distances(evalEmails, trainingEmails)

        # get prediction for each email
        result = []
        for email in range(len(evalSet)):
            # get class of nearest neighbor to current email
            result.append(int(trainingSet[np.argmin(distances[email])][-1]))

        return result

