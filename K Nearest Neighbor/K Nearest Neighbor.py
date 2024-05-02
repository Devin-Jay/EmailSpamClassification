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

    KNNClass = KNearestNeighbor(dataset[0])

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
        KNNClass.classifySet(trainingSet[fold:fold + len(trainingSet) // K], trainingSet)

class KNearestNeighbor:
    def __init__(self, attributes):
        self.labels = attributes
        self.numOfXVals = len(self.labels) - 1
    
    def evalEmail(self, evalEmail, trainingSet):
        # calculate |V| for evalEmail
        V1 = 0
        for attribute in range(self.numOfXVals):
            V1 += float(evalEmail[attribute]) ** 2
        V1 = math.sqrt(V1)

        # index 0 is cosine similarity; index 1 is email's class
        cosSimilarity = (0,0)
        for email in trainingSet:
            numerator = 0
            V2 = 0
            for attribute in range(self.numOfXVals):
                numerator += (float(evalEmail[attribute]) * float(email[attribute]))
                V2 += float(email[attribute]) ** 2
            
            emailSimilarity = numerator / (V1 * math.sqrt(V2))

            if (emailSimilarity > cosSimilarity[0]):
                cosSimilarity = (emailSimilarity, email[-1])
        
        return cosSimilarity
            


    def classifySet(self, evalSet, trainingSet):
        result = []
        for email in range(len(evalSet)):
            result.append(self.evalEmail(evalSet[email], trainingSet))
        print(result)

main("spambase.csv",0.8)