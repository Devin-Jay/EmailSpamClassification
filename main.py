import math
import random

import Performance as perf
from NaiveBayes.NaiveBayes import NaiveBayesModel
from LogisticRegression.LogisticRegression import LogisticRegression
from KNearestNeighbor.KNearestNeighbor import KNearestNeighbor

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

    # initialize model classes with data labels
    NBAlgorithm = NaiveBayesModel(dataset[0])
    LRClass = LogisticRegression(dataset[0])
    KNNClass = KNearestNeighbor(dataset[0])

    #shuffle dataset
    dataset = dataset[1:]
    random.shuffle(dataset)
    
    # get training set
    trainingSize = math.floor(trainingPct * (len(dataset)))
    trainingSet = dataset[:trainingSize + 1]
    random.shuffle(trainingSet)

    # get evalSet and evalSet actualvals
    evalSet = dataset[trainingSize:]
    evalActualResults = [int(email[-1]) for email in evalSet]

    foldResults = []
    evalResults = []
    allModelResults = []
    
    ### NAIVE BAYES ###
    # loop through training (increment by size of training // 5 for 5 cross validation)
    for fold in range(0, len(trainingSet) - (len(trainingSet) // K), len(trainingSet) // K):
        # train Naive Bayes model and test on fold
        predictedResults, spamPrior, hamPrior, spamLikelihoods, hamLikelihoods = NBAlgorithm.naiveBayes(fold, fold + len(trainingSet) // K, trainingSet)

        # get actualResults
        actualResults = [int(email[-1]) for email in trainingSet[fold:fold + len(trainingSet) // K]]

        # get performance results
        acc, fp, tp, auc = perf.performance(predictedResults, actualResults)

        # add to fold results
        foldResults.append((acc, fp, tp, auc))

        # test on evalSet
        result = NBAlgorithm.useModel(spamPrior, hamPrior, spamLikelihoods, hamLikelihoods, evalSet)

        # get performance results on evalSet
        acc, fp, tp, auc = perf.performance(result, evalActualResults)

        # add to evalResults
        evalResults.append((acc, fp, tp, auc))

    allModelResults.append(perf.average(foldResults, K))
    allModelResults.append(perf.average(evalResults, K))

    foldResults = []
    evalResults = []

    ### Linear Regression ###
    # loop through training (increment by size of training // 5 for 5 cross validation)
    for fold in range(0, len(trainingSet) - (len(trainingSet) // K), len(trainingSet) // K):
        # train Linear Regression model and test on fold
        predictedResults, model = LRClass.LRAlgorithm(fold, fold + len(trainingSet) // K, trainingSet)

        # get actualResults
        actualResults = [int(email[-1]) for email in trainingSet[fold:fold + len(trainingSet) // K]]

        # get performance results
        acc, fp, tp, auc = perf.performance(predictedResults, actualResults)

        # add to fold results
        foldResults.append((acc, fp, tp, auc))

        # test on evalSet
        result = LRClass.useModel(model, evalSet)

        # get performance results on evalSet
        acc, fp, tp, auc = perf.performance(result, evalActualResults)

        # add to evalResults
        evalResults.append((acc, fp, tp, auc))
    
    allModelResults.append(perf.average(foldResults, K))
    allModelResults.append(perf.average(evalResults, K))

    foldResults = []
    evalResults = []

    ### K Nearest Neighbor ###
    # loop through training (increment by size of training // 5 for 5 cross validation)
    for fold in range(0, len(trainingSet) - (len(trainingSet) // K), len(trainingSet) // K):
        # get emails to compare to
        compareEmails = trainingSet[:fold] + trainingSet[fold + len(trainingSet) // K:]

        # test on fold
        predictedResults = KNNClass.classifySet(trainingSet[fold:fold + len(trainingSet) // K], compareEmails)

        # get actualResults
        actualResults = [int(email[-1]) for email in trainingSet[fold:fold + len(trainingSet) // K]]

        # get performance results
        acc, fp, tp, auc = perf.performance(predictedResults, actualResults)

        # add to fold results
        foldResults.append((acc, fp, tp, auc))

        # test on evalSet
        result = KNNClass.classifySet(evalSet, compareEmails)

        # get performance results on evalSet
        acc, fp, tp, auc = perf.performance(result, evalActualResults)

        # add to evalResults
        evalResults.append((acc, fp, tp, auc))
    
    allModelResults.append(perf.average(foldResults, K))
    allModelResults.append(perf.average(evalResults, K))

    print(allModelResults)

main("spambase.csv",0.8)