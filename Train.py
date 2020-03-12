from MLP import *
from math import sqrt, exp
import numpy as np
import random

#import the data
def importData(path):
    file = open(path, 'r', encoding='utf-8-sig')
    lines = file.readlines()
    file.close()
    #process the csv file to return an array
    strData = []
    for i in lines:
        i = i.strip("\n")
        strData.append(i.split(","))

    floatData = []
    for i in strData:
        data = []
        for j in i:
            data.append(float(j))
        floatData.append(data)

    return floatData

def separateData(dataSet, trainingSize, validationSize, testSize):
    data = dataSet
    training = []
    for i in range(trainingSize):
        randIndex = random.randrange(len(data))
        training.append(data[randIndex])
        data = data[0:randIndex] + data[randIndex+1:]
    validation = []
    for i in range(validationSize):
        randIndex = random.randrange(len(data))
        validation.append(data[randIndex])
        data = data[0:randIndex] + data[randIndex+1:]
    test = []
    for i in range(testSize):
        randIndex = random.randrange(len(data))
        test.append(data[randIndex])
        data = data[0:randIndex] + data[randIndex+1:]
    return training, validation, test

def epoch(mlp, rho, trainingData):
    error = []
    for line in trainingData:
        t = line[0]
        w = line[1]
        sr = line[2]
        dsp = line[3]
        drh = line[4]
        panE = line[5]

        mlp.setInputs([t,w,sr,dsp,drh])
        error.append(panE-mlp.forwardPass()[0])
        mlp.backwardPass(panE)
        mlp.updateWeights(rho)

    return error


def RMSE(error):
    squaredError = []
    for i in error:
        squaredError.append(i**2)
    return sqrt(sum(squaredError)/len(squaredError))

def validate(mlp, validationData):
    error = []
    for line in validationData:
        t = line[0]
        w = line[1]
        sr = line[2]
        dsp = line[3]
        drh = line[4]
        panE = line[5]

        mlp.setInputs([t,w,sr,dsp,drh])
        error.append(panE-mlp.forwardPass()[0])

    return error

def test(mlp, testData):
    error = []
    for line in testData:
        t = line[0]
        w = line[1]
        sr = line[2]
        dsp = line[3]
        drh = line[4]
        panE = line[5]

        mlp.setInputs([t,w,sr,dsp,drh])
        error.append(panE-mlp.forwardPass()[0])

    return error

def annealingParam(start, end, max, current):
    return end + (start-end)*(1-1/(1+exp(10-(20*current/max))))


def trainMLP(mlp, rho, trainingData, validationData, testData, outputPath, comment, annealing=False, epochLimit=-1, startParam=-1, endParam=-1):
    network = mlp
    file = open(outputPath, 'a')
    print(comment)
    file.write(comment+'\n')
    if annealing == False:
        stop = False
        i=0
        while stop == False:
            epochError = RMSE(epoch(network, rho, trainingData))
            if i == 0:
                lastValidError = RMSE(validate(network, validationData))
                fileString = str(lastValidError)+","+str(epochError)+"\n"
                file.write(fileString)
                print("Epoch %d RMSE: %f, Validation: %f"%(i,epochError,lastValidError))
            elif i%10 == 0 and i != 0:
                validationError = RMSE(validate(network, validationData))
                fileString = str(validationError)+","+str(epochError)+"\n"
                file.write(fileString)
                print("Epoch %d RMSE: %f, Validation: %f"%(i,epochError,validationError))
                if lastValidError < validationError:
                    stop = True
                    testError = RMSE(test(network, testData))
                    fileString = "Test RMSE: "+str(testError)+"\n"
                    file.write(fileString)
                    print(testError)
                else:
                    lastValidError = validationError
            i +=1
    else:
        i = 0
        while i <= epochLimit:
            learnParam = annealingParam(startParam, endParam, epochLimit, i)
            epochError = RMSE(epoch(network, learnParam, trainingData))
            if i%10 == 0:
                validationError = RMSE(validate(network, validationData))
                fileString = str(validationError)+","+str(epochError)+"\n"
                file.write(fileString)
                print("Epoch %d RMSE: %f, Validation: %f"%(i,epochError,validationError))
            i+=1
        testError = RMSE(test(network, testData))
        fileString = "Test RMSE: "+str(testError)+"\n"
        file.write(fileString)
        print(testError)
    file.close()
    print("Done")

dataSet = importData("data/fullset.csv")
trainingData, validationData, testData = separateData(dataSet, 866, 288, 289)

ann = MLP(4, [5,3,2,1])

trainMLP(ann, 0.01, trainingData, validationData, testData, "results/test4.csv", "2 hidden layers (3,2) rho 0.01", annealing=False, epochLimit=1000, startParam=0.01, endParam=0.005)
