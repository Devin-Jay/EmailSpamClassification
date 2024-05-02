import math
import random

K = 5

def importDataset(fileName):
    with open(fileName) as file:
        dataset = file.readlines()
        for email in range(len(dataset)):
            dataset[email] = dataset[email].split(",")
    return dataset

def main(fileName, trainingPct):

    # get dataset
    dataset = importDataset("spambase.csv")

    LRClass = LogisticRegression(dataset[0])

    #shuffle dataset
    dataset = dataset[1:]
    random.shuffle(dataset)

    # get training set
    trainingSize = math.floor(trainingPct * (len(dataset)))
    trainingSet = dataset[:trainingSize + 1]
    random.shuffle(trainingSet)

    foldResults = []
    
    # loop through training (increment by size of training // 5 for 5 cross validation)
    for fold in range(0, len(trainingSet) - (len(trainingSet) // K), len(trainingSet) // K):
    #for fold in range(0, (len(trainingSet) // K), len(trainingSet) // K):
        # evaluate fold
        LRClass.LRAlgorithm(fold, fold + len(trainingSet) // K, trainingSet)
    
    print()
    print(foldResults)
    #wholeDatasetAverage = NBAlgorithm.naiveBayes(3680,len(dataset), dataset)
    #print(wholeDatasetAverage)


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
        linearYVal += model[attribute]

        # return sigmoid y val
        return 1 / (1 + math.exp(-linearYVal))

    def gradientDescent(self, trainingSet, learningRate, iterations):
        # initialize weights with random num between 0 and 3 for each attribute (plus 1 for bias)
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
            
            # update biasS
            gradient = (2/len(trainingSet)) * (predY - yVal) * actualWeights[-1]
            actualWeights[-1] -= (learningRate * gradient)
        
        return actualWeights

    def LRAlgorithm(self, startIndex, endIndex, dataset):
        # get training set
        trainingSet = dataset[:startIndex] + dataset[endIndex + 1:len(dataset)]

        weights = self.gradientDescent(trainingSet, 0.001, 100)

        for email in range(startIndex, endIndex):
            y = self.sigmoid(weights, dataset[email][:-1])
            if (y < 0.5):
                result = 1
            else:
                result = 0
            if (result == int(dataset[email][-1])):
                print("yes")
            else:
                print("no")


main("spambase.csv",0.8)