from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

def performance(predictedResults, actualResults):

    # calculate accuracy score
    acc = accuracy_score(actualResults, predictedResults)

    # calculate false positives rate
    fp = findFP(predictedResults, actualResults)

    # calculate true positive rate
    tp = findTP(predictedResults, actualResults)

    # calculate AUC
    auc = roc_auc_score(actualResults, predictedResults)

    return acc, fp, tp, auc

def average(results, K):
    accAvg = 0
    fpAvg = 0
    tpAvg = 0
    aucAvg = 0

    for result in results:
        accAvg += result[0]
        fpAvg += result[1]
        tpAvg += result[2]
        aucAvg += result[3]
    
    accAvg /= K
    fpAvg /= K 
    tpAvg /= K
    aucAvg /= K

    return (accAvg,fpAvg, tpAvg, aucAvg)

def findFP(predictedResults, actualResults):
    # fp = fp / (fp + tn)
    # get number of false positives (ham labeled spam) and number of true negatives (spam labeled ham)
    fp = 0
    tn = 0
    for classifier in range(len(actualResults)):
        if (predictedResults[classifier] and not actualResults[classifier]):
            fp += 1
        if (not predictedResults[classifier] and actualResults[classifier]):
            tn += 1
    
    if (not fp):
        return 0

    return fp / (fp + tn)

def findTP(predictedResults, actualResults):
    # tp = tp / (tp + fn)
    # get number of true positives (spam labeled spam) and number of false negatives (ham labeled ham)
    tp = 0
    fn = 0
    for classifier in range(len(actualResults)):
        if (predictedResults[classifier] and actualResults[classifier]):
            tp += 1
        if (not predictedResults[classifier] and not actualResults[classifier]):
            fn += 1
    
    if (not tp):
        return 0

    return tp / (tp + fn)