import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import math as ma
import random

#Sigmoid function
def sigmoid(x):
    return 1/(np.exp(-x)+1)

#derivative of sigmoid
def d_sigmoid(x):
    return (np.exp(-x))/((np.exp(-x)+1)**2)

#Softmax
def softmax(x):
    exp_element = np.exp(x-x.max())
    return exp_element/np.sum(exp_element,axis=0)

#derivative of softmax
def d_softmax(x):
    exp_element = np.exp(x-x.max())
    botVal = np.sum(exp_element,axis=0)
    botVal[botVal == 0] = 0.001
    test1 = (1-exp_element/botVal)
    test2 = exp_element/botVal
    return test2*test1


def TrainingTheNetwork():
    trainDataSet = np.loadtxt("trainDataSet.txt", delimiter=',')
    trainLabels = np.loadtxt("trainLabels.txt", delimiter=',')
    testDataSet = np.loadtxt("testDataSet.txt", delimiter=',')
    testLabels = np.loadtxt("testLabels.txt", delimiter=',')
    wInit = np.loadtxt("InitialWeights.txt", delimiter=',')
    wInitChanged = wInit
    wHidden = np.loadtxt("InitialHiddenWeights.txt", delimiter=',')
    wHiddenChanged = wHidden
    print("Training Started!")
    epochs = 3
    n = 0.01
    previousError = 0
    accuracies = []
    for i in range(0, epochs):
        error = 0
        iterator = 0
        curwInit = wInitChanged
        curwHidden = wHiddenChanged
        print("Epoch :", i)

        # Hidden Layer TEST
        tempArraytest = np.dot(testDataSet, curwInit)
        hiddenSigmoidtest = sigmoid(tempArraytest)

        # Output Layer TEST
        outputNueronValuestest = np.dot(hiddenSigmoidtest, wHidden)
        outputSoftMaxtest = softmax(outputNueronValuestest)

        #Hidden Layer
        tempArray = np.dot(trainDataSet, curwInit)
        hiddenSigmoid = sigmoid(tempArray)

        #Output Layer
        outputNueronValues = np.dot(hiddenSigmoid, wHidden)
        outputSoftMax = softmax(outputNueronValues)

        S_iValue = []
        for eachNum in range(0,10):

            S_iValue = (((outputSoftMax[:,eachNum] - trainLabels)*(d_softmax(tempArray.T))).T)
            wHiddenChanged[:,eachNum] = curwHidden[:,eachNum] + ((n*((((outputSoftMax[:,eachNum] - trainLabels)*(d_softmax(tempArray.T))).T)*hiddenSigmoid)).sum(axis=0))

            s_jValue = ((d_sigmoid(tempArray)).T * (np.dot(S_iValue, curwHidden[:,eachNum]))).T
            for initialWeighsNum in range(0,150):
                wInitChanged[:,initialWeighsNum] = curwInit[:,initialWeighsNum] + (((n*s_jValue[:,initialWeighsNum])*trainDataSet.T).T.sum(axis=0))
            # print("works")
            # print(np.size(wInitChanged, 0))
            # print(np.size(wInitChanged, 1))

        category = np.argmax(outputSoftMax, axis=1)
        categoryTest = np.argmax(outputSoftMaxtest, axis=1)
        accuracy = [0,0]
        accuracy[0] = (category == trainLabels).mean()
        accuracy[1] = (categoryTest == testLabels).mean()
        accuracies.append(accuracy)

        print("Accuracies: ", accuracies)
        print("_________________________________")


    # print("testing")
    # print(np.size(outputSoftMax, 0))
    # print(np.size(outputSoftMax, 1))
    print("testing1")
    print(np.size(outputSoftMaxtest, axis=1))
    print(np.size(outputSoftMaxtest, axis=0))
    # print("testing2")
    # print(np.size(trainLabels, 0))
    # print("testing3")
    # print(np.size(testLabels, 0))
    confusionTrain = confusion_matrix(np.argmax(outputSoftMax, axis=1), np.argmax(trainLabels,axis=0))
    confusionTest = confusion_matrix(np.argmax(outputSoftMaxtest, axis=1), np.argmax(testLabels, axis=0))
    np.savetxt('ConMatTrain.txt', confusionTrain, fmt='%1.4f', delimiter=',')
    np.savetxt('ConMatTest.txt', confusionTest, fmt='%1.4f', delimiter=',')
    np.savetxt('Accuracies.txt', accuracies, fmt='%1.4f', delimiter=',')
    np.savetxt('TrainedWeights.txt', wInitChanged, fmt='%1.4f', delimiter=',')
    np.savetxt('TrainedHiddenWeights.txt', wHiddenChanged, fmt='%1.4f', delimiter=',')
    print("Training Finished!")


def test():
    trainDataSet = np.loadtxt("testDataSet.txt", delimiter=',')
    trainLabels = np.loadtxt("testLabels.txt", delimiter=',')
    wInit = np.loadtxt("TrainedWeights.txt", delimiter=',')
    wHidden = np.loadtxt("TrainedHiddenWeights.txt", delimiter=',')
    wInitChanged = wInit
    wHiddenChanged = wHidden
    print("Training Started!")
    epochs = 2
    n = 0.01
    previousError = 0
    accuracies = []
    for i in range(0, epochs):
        error = 0
        iterator = 0
        curwInit = wInitChanged
        curwHidden = wHiddenChanged
        print("Epoch :", i)

        # Hidden Layer
        tempArray = np.dot(trainDataSet, curwInit)
        hiddenSigmoid = sigmoid(tempArray)

        # Output Layer
        outputNueronValues = np.dot(hiddenSigmoid, wHidden)
        outputSoftMax = softmax(outputNueronValues)

        S_iValue = []
        for eachNum in range(0, 10):

            S_iValue = (((outputSoftMax[:, eachNum] - trainLabels) * (d_softmax(tempArray.T))).T)
            wHiddenChanged[:, eachNum] = curwHidden[:, eachNum] + (
                (n * ((((outputSoftMax[:, eachNum] - trainLabels) * (d_softmax(tempArray.T))).T) * hiddenSigmoid)).sum(
                    axis=0))

            s_jValue = ((d_sigmoid(tempArray)).T * (np.dot(S_iValue, curwHidden[:, eachNum]))).T
            for initialWeighsNum in range(0, 150):
                wInitChanged[:, initialWeighsNum] = curwInit[:, initialWeighsNum] + (
                    ((n * s_jValue[:, initialWeighsNum]) * trainDataSet.T).T.sum(axis=0))
            # print("works")
            # print(np.size(wInitChanged, 0))
            # print(np.size(wInitChanged, 1))
        print("testing")
        category = np.argmax(outputSoftMax, axis=1)
        accuracy = (category == trainLabels).mean()
        accuracies.append(accuracy)

        print("Accuracies: ", accuracies)
        print("_________________________________")
    print("Training Finished!")

def plotAccuracy():
    Accuracies = np.loadtxt("Accuracies.txt", delimiter=',')
    # AccuraciesTest = np.loadtxt("AccuraciesTest.txt", delimiter=',')

    fig, (ax1) = plt.subplots(1)
    ax1.plot(range(50), Accuracies, label="Accuracy")
    #ax1.plot(range(40), AccuraciesTest, label="Accuracy Test")
    ax1.set_title("Training and test error")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Error Percentage")
    ax1.set_xlim(0, 50)
    ax1.set_ylim(0, 1)

    plt.legend()
    plt.show()
