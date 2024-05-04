import random
import math

class LogisticRegression:
    def __init__(self, attributeLabels):
        self.labels = attributeLabels
        self.numOfXVals = len(self.labels) - 1
    
    def getXVals(self, trainingSet):
        # add one more for bias
        xVals = [0] * self.numOfXVals
        
        for email in range(len(trainingSet)):
            for attribute in range(self.numOfXVals):
                xVals[attribute] += float(trainingSet[email][attribute])
        
        trainingSetSize = len(trainingSet)
        for attribute in range(self.numOfXVals):
            xVals[attribute] /= trainingSetSize
        
        return xVals
    
    def getYVal(self, trainingSet):
        yVal = 0
        for email in range(len(trainingSet)):
            yVal += float(trainingSet[email][-1])
        
        yVal /= len(trainingSet)
        
        return yVal
    
    def sigmoid(self, model, xVals):
        linearYVal = 0
        for attribute in range(self.numOfXVals):
            linearYVal += model[attribute] * float(xVals[attribute])
        # add the bias
        linearYVal += model[-1]

        # return sigmoid y val
        return 1 / (1 + math.exp(-linearYVal))

    def gradientDescent(self, trainingSet, learningRate, iterations):
        # initialize weights with random num between 0 and 1 for each attribute (plus 1 for bias)
        actualWeights = [random.uniform(0,1) for attribute in range(self.numOfXVals + 1)]

        xVals = self.getXVals(trainingSet)
        yVal = self.getYVal(trainingSet)

        for i in range(iterations):
            # get predicted y
            predY = self.sigmoid(actualWeights, xVals)

            for attribute in range (self.numOfXVals):
                # calculate the gradient of attribute
                gradient = (2/len(trainingSet)) * (predY - yVal) * xVals[attribute]

                # update actual weights
                actualWeights[attribute] -= (learningRate * gradient)
            
            # update bias
            gradient = (2/len(trainingSet)) * (predY - yVal) * actualWeights[-1]
            actualWeights[-1] -= (learningRate * gradient)
        
        return actualWeights
    
    def useModel(self, weights, evalSet):
        emailPredictions = []

        for email in evalSet:
            prediction = 0
            for attribute in range(len(email)):
                prediction += weights[attribute] * float(email[attribute])
            
            if (prediction > 0.5):
                emailPredictions.append(1)
            else:
                emailPredictions.append(0)

        return emailPredictions

    def LRAlgorithm(self, startIndex, endIndex, dataset):
        # get training set
        trainingSet = dataset[:startIndex] + dataset[endIndex + 1:len(dataset)]

        # fit logistic regression line to graph
        weights = self.gradientDescent(trainingSet, 0.001, 1000)

        # test on fold
        results = self.useModel(weights, dataset[startIndex:endIndex])

        return results, weights