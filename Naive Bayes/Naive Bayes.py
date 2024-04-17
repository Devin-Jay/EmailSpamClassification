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
    random.shuffle(dataset[1:])
    
    # get training set
    trainingSize = math.floor(trainingPct * (len(dataset)))
    trainingSet = dataset[1:trainingSize + 1]
    random.shuffle(trainingSet)

    #  Get eval set
    TestSet = dataset[trainingSize:]

    foldResults = []

    NBAlgorithm = NaiveBayesModel(dataset[0])
    
    # loop through training (increment by size of training // 5 for 5 cross validation)
    for fold in range(0, len(trainingSet) - (len(trainingSet) // K), len(trainingSet) // K):
        # evaluate fold
        foldResults.append(NBAlgorithm.naiveBayes(fold+1, fold + len(trainingSet) // K, trainingSet))
    
    print()
    print(foldResults)
    wholeDatasetAverage = NBAlgorithm.naiveBayes(3680,len(dataset), dataset[1:])
    print(wholeDatasetAverage)

class NaiveBayesModel:
    def __init__(self, attributeLabels):
        self.labels = attributeLabels

    # function that returns list of emails that are classified as yes or no (depends on parameter input)
    def getClassifiedData(self, dataset, spam):
        classifiedData = []

        for email in range(1, len(dataset)):
            if (int(dataset[email][-1]) == spam):
                classifiedData.append(dataset[email])

        return classifiedData

    def getNumOf(self, target, dataset):
        count = 0
        for column in range(len(dataset[0])):
            if (target in dataset[0][column]):
                count += 1
        return count
    
    def calculateLikelihoods(self, dataset):
        # get data that was classified as spam
        spamData = self.getClassifiedData(dataset, True)
        spamDataSize = len(spamData)

        # get data that was classified as normal
        hamData = self.getClassifiedData(dataset, False)
        hamDataSize = len(hamData)

        # initialize counts with 1 (laplace smoothing)
        spamLikelihoods = [1] * (len(self.labels) - 4)
        hamLikelihoods = [1] * (len(self.labels) - 4)

        totalSpamWords = len(self.labels) - 4
        totalHamWords = len(self.labels) - 4

        for attribute in range(len(self.labels) - 4):
            for email in range(1, spamDataSize):
                if (float(spamData[email][attribute]) > 0.0):
                    spamLikelihoods[attribute] += 1
                    totalSpamWords += 1
            
            for email in range(1, hamDataSize):
                if (float(hamData[email][attribute]) > 0.0):
                    hamLikelihoods[attribute] += 1
                    totalHamWords += 1

        for attribute in range(len(self.labels) - 4):
            spamLikelihoods[attribute] /= totalSpamWords
            hamLikelihoods[attribute] /= totalHamWords
        
        spamPrior = spamDataSize / (spamDataSize + hamDataSize)
        hamPrior = hamDataSize / (spamDataSize + hamDataSize)

        return spamPrior, hamPrior, spamLikelihoods, hamLikelihoods
    
    def useModel(self, spamPrior, hamPrior, spamLikelihoods, hamLikelihoods, evalSet):
        emailPredictions = []
        print(spamPrior, hamPrior)
        for email in range(len(evalSet)):
            emailsSpamLikelihoods = 1
            emailsHamLikelihoods = 1

            for attribute in range(len(self.labels) - 4):
                if (float(evalSet[email][attribute]) > 0.0):
                    emailsSpamLikelihoods *= float(evalSet[email][attribute])#spamLikelihoods[attribute]#
                    emailsHamLikelihoods *= float(evalSet[email][attribute])#hamLikelihoods[attribute]#

            emailsSpamLikelihoods *= spamPrior
            emailsHamLikelihoods *= hamPrior

            if (emailsSpamLikelihoods > emailsHamLikelihoods):
                emailPredictions.append(1)
            else:
                emailPredictions.append(0)
        
        return emailPredictions
    
    def evaluateModel(self, result, evalSet):
        count = 0
        for email in range(len(evalSet)):
            if (result[email] == int(evalSet[email][-1])):
                count += 1
        average = count / len(evalSet)
        return average

    def naiveBayes(self, evalStartIndex, evalEndIndex, dataset):
        # initialize training set (all other folds, not current fold)
        trainingSet = dataset[evalEndIndex:len(dataset)] + dataset[:evalStartIndex]

        # get likelihoods and priors
        spamPrior, hamPrior, spamLikelihoods, hamLikelihoods = self.calculateLikelihoods(trainingSet)
        
        # get dataset to be tested on (current fold)
        evalSet = dataset[evalStartIndex:evalEndIndex]

        # test current model on current fold
        result = self.useModel(spamPrior, hamPrior, spamLikelihoods, hamLikelihoods, evalSet)

        evaluation = self.evaluateModel(result, evalSet)
        return evaluation
        

        

main("spambase.csv", 0.8)
