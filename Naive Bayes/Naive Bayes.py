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

    # initialize NaiveBayesModel class with data labels
    NBAlgorithm = NaiveBayesModel(dataset[0])

    #shuffle dataset
    dataset = dataset[1:]
    random.shuffle(dataset)
    
    # get training set
    trainingSize = math.floor(trainingPct * (len(dataset)))
    trainingSet = dataset[:trainingSize + 1]
    random.shuffle(trainingSet)

    # get evalSet
    evalSet = dataset[trainingSize:]

    foldResults = []
    foldPriosAndLikelihoods = []
    
    # loop through training (increment by size of training // 5 for 5 cross validation)
    for fold in range(0, len(trainingSet) - (len(trainingSet) // K), len(trainingSet) // K):
        # evaluate fold
        evaluation, spamPrior, hamPrior, spamLikelihoods, hamLikelihoods = NBAlgorithm.naiveBayes(fold, fold + len(trainingSet) // K, trainingSet)
        foldResults.append(evaluation)
        foldPriosAndLikelihoods.append((spamPrior, hamPrior, spamLikelihoods, hamLikelihoods))
    
    # get best model from folds
    bestModel = 0
    for fold in range(len(foldResults)):
        if (foldResults[fold] > bestModel):
            bestModel = fold
    
    bestModel = foldPriosAndLikelihoods[bestModel]

    # test current model on current fold
    result = NBAlgorithm.useModel(bestModel[0], bestModel[1], bestModel[2], bestModel[3], evalSet)

    # evaluate model
    evaluation = NBAlgorithm.evaluateModel(result, evalSet)
    

class NaiveBayesModel:
    def __init__(self, attributeLabels):
        self.labels = attributeLabels

    # function that returns list of emails that are classified as yes or no (depends on parameter input)
    def getClassifiedData(self, dataset, spam):
        classifiedData = []

        for email in range(len(dataset)):
            if (int(dataset[email][-1]) == spam):
                classifiedData.append(dataset[email])

        return classifiedData
    
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

        for email in range(len(evalSet)):
            emailsSpamLikelihoods = 1
            emailsHamLikelihoods = 1

            for attribute in range(len(self.labels) - 4):
                if (float(evalSet[email][attribute]) > 0.0):
                    emailsSpamLikelihoods *= spamLikelihoods[attribute]
                    emailsHamLikelihoods *= hamLikelihoods[attribute]
            
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
        trainingSet = dataset[:evalStartIndex] + dataset[evalEndIndex + 1:len(dataset)]

        # get likelihoods and priors
        spamPrior, hamPrior, spamLikelihoods, hamLikelihoods = self.calculateLikelihoods(trainingSet)

        # get dataset to be tested on (current fold)
        evalSet = dataset[evalStartIndex:evalEndIndex]

        # test current model on current fold
        result = self.useModel(spamPrior, hamPrior, spamLikelihoods, hamLikelihoods, evalSet)

        # evaluate model
        evaluation = self.evaluateModel(result, evalSet)

        return evaluation, spamPrior, hamPrior, spamLikelihoods, hamLikelihoods
        

        

main("spambase.csv", 0.8)
