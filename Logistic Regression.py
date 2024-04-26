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
    #for fold in range(0, len(trainingSet) - (len(trainingSet) // K), len(trainingSet) // K):
    for fold in range(0, (len(trainingSet) // K), len(trainingSet) // K):
        # evaluate fold
        #foldResults.append(LRClass.LRAlgorithm(fold, fold + len(trainingSet) // K, trainingSet))
        LRClass.LRAlgorithm(fold, fold + len(trainingSet) // K, trainingSet)
    
    print()
    print(foldResults)
    #wholeDatasetAverage = NBAlgorithm.naiveBayes(3680,len(dataset), dataset)
    #print(wholeDatasetAverage)


class LogisticRegression:
    def __init__(self, attributeLabels):
        self.labels = attributeLabels
        self.numOfXVals = len(self.labels) - 1
    
    def gradientDescent(self, trainingSet, learningRate, iterations):
        # initialize weights with random num between 0 and 3 for each attribute (plus 1 for bias)
        actualWeights = [random.uniform(0,3) for attribute in range(self.numOfXVals + 1)]

        for i in range(iterations):
            
            for attribute in range (self.numOfXVals):
                pass

    def LRAlgorithm(self, startIndex, endIndex, dataset):
        # get training set
        trainingSet = dataset[:startIndex] + dataset[endIndex + 1:len(dataset)]

        self.gradientDescent(trainingSet, 0.01, 500)

main("spambase.csv",0.8)