import random
import math
import numpy as np
from sklearn.metrics import accuracy_score

class LogisticRegression:
    def __init__(self, attributeLabels):
        self.labels = attributeLabels
        self.numOfXVals = len(self.labels) - 1
    
    def getYVal(self, trainingSet):
        yVal = []
        for email in range(len(trainingSet)):
            yVal.append(int(trainingSet[email][-1]))
        
        return yVal
    
    def sigmoid2(self, x):
        clippedX = np.clip(x,-500,500)
        return 1 / (1 + np.exp(-clippedX))

    def gradientDescent(self, trainingSet, learningRate, iterations):
        # get classifiers of all emails in trainingSet
        yVal = np.array(self.getYVal(trainingSet))

        # get trainingSet without target column
        xWithBias = []
        for email in trainingSet:
            floatEmail =  [round(float(x),2) for x in email]
            xWithBias.append([0] + floatEmail)
        xWithBias = np.array(xWithBias)

        # get len of dataset
        m, n = xWithBias.shape

        # get model initialized with 0s
        model =  np.zeros(n)

        # fit line
        for i in range(iterations):
            predY = self.sigmoid2(np.dot(xWithBias,model))
            gm = np.dot(xWithBias.T, (predY - yVal)) / m
            model -= (learningRate * gm)

        return model

    def useModel(self, weights, evalSet):
        emailPredictions = []

        for email in evalSet:
            prediction = 0
            for attribute in range(1,len(email) - 1):
                prediction += weights[attribute] * float(email[attribute])
            
            prediction += weights[0]

            if (prediction > 0.5):
                emailPredictions.append(1)
            else:
                emailPredictions.append(0)

        return emailPredictions

    def LRAlgorithm(self, startIndex, endIndex, dataset):
        # get training set
        trainingSet = dataset[:startIndex] + dataset[endIndex + 1:len(dataset)]

        # fit logistic regression line to graph
        weights = self.gradientDescent(trainingSet, 0.000001, 1000)

        # test on fold
        results = self.useModel(weights, dataset[startIndex:endIndex])

        return results, weights