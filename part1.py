import numpy as np
import matplotlib.pyplot as plt
import random


def setCreator():
    train_dataImg = np.loadtxt("MNISTnumImages5000_balanced.txt")
    train_dataLabel = np.loadtxt("MNISTnumLabels5000_balanced.txt")
    zero = []
    one = []
    two = []
    three = []
    four = []
    five = []
    six = []
    seven = []
    eight = []
    nine = []
    interator = 0
    # print(train_dataImg.shape)
    for num in train_dataImg:
        # (train_dataImg[interator])
        if train_dataLabel[interator] == 0:
            zero.append(train_dataImg[interator])
        if train_dataLabel[interator] == 1:
            one.append(train_dataImg[interator])
        if train_dataLabel[interator] == 2:
            two.append(train_dataImg[interator])
        if train_dataLabel[interator] == 3:
            three.append(train_dataImg[interator])
        if train_dataLabel[interator] == 4:
            four.append(train_dataImg[interator])
        if train_dataLabel[interator] == 5:
            five.append(train_dataImg[interator])
        if train_dataLabel[interator] == 6:
            six.append(train_dataImg[interator])
        if train_dataLabel[interator] == 7:
            seven.append(train_dataImg[interator])
        if train_dataLabel[interator] == 8:
            eight.append(train_dataImg[interator])
        if train_dataLabel[interator] == 9:
            nine.append(train_dataImg[interator])
        interator = interator + 1

    print(interator)
    random.shuffle(zero)
    random.shuffle(one)
    random.shuffle(two)
    random.shuffle(three)
    random.shuffle(four)
    random.shuffle(five)
    random.shuffle(six)
    random.shuffle(seven)
    random.shuffle(eight)
    random.shuffle(nine)
    np.savetxt('zeroSet.txt', zero, fmt='%1.4f', delimiter=',')
    np.savetxt('oneSet.txt', one, fmt='%1.4f', delimiter=',')
    np.savetxt('twoSet.txt', two, fmt='%1.4f', delimiter=',')
    np.savetxt('threeSet.txt', three, fmt='%1.4f', delimiter=',')
    np.savetxt('fourSet.txt', four, fmt='%1.4f', delimiter=',')
    np.savetxt('fiveSet.txt', five, fmt='%1.4f', delimiter=',')
    np.savetxt('sixSet.txt', six, fmt='%1.4f', delimiter=',')
    np.savetxt('sevenSet.txt', seven, fmt='%1.4f', delimiter=',')
    np.savetxt('eightSet.txt', eight, fmt='%1.4f', delimiter=',')
    np.savetxt('nineSet.txt', nine, fmt='%1.4f', delimiter=',')

def CreateTrainSet():
    totalIterator = 0
    oneCount = 0
    zeroCount = 0
    twoCount = 0
    threeCount = 0
    fourCount = 0
    fiveCount = 0
    sixCount = 0
    sevenCount = 0
    eightCount = 0
    nineCount = 0

    # Iterators to keep track of number for test and train sets respectively
    oneIterator = 0
    zeroIterator = 0
    twoIterator = 0
    threeIterator = 0
    fourIterator = 0
    fiveIterator = 0
    sixIterator = 0
    sevenIterator = 0
    eightIterator = 0
    nineIterator = 0

    oneSet = np.loadtxt("oneSet.txt", delimiter=",")
    zeroSet = np.loadtxt("zeroSet.txt", delimiter=",")
    twoSet = np.loadtxt("twoSet.txt", delimiter=",")
    threeSet = np.loadtxt("threeSet.txt", delimiter=",")
    fourSet = np.loadtxt("fourSet.txt", delimiter=",")
    fiveSet = np.loadtxt("fiveSet.txt", delimiter=",")
    sixSet = np.loadtxt("sixSet.txt", delimiter=",")
    sevenSet = np.loadtxt("sevenSet.txt", delimiter=",")
    eightSet = np.loadtxt("eightSet.txt", delimiter=",")
    nineSet = np.loadtxt("nineSet.txt", delimiter=",")
    trainLabels = []
    testLabels = []
    testDataSet = []
    trainDataSet = []

    for num in range(5000):
        whatValue = np.random.randint(10)
        validValueCheck = True
        while validValueCheck:
            #Check all possible values to make sure random generated value is possible
            if (whatValue == 1) and (oneCount == 500):
                validValueCheck = True
            elif (whatValue == 0) and (zeroCount == 500):
                validValueCheck = True
            elif (whatValue == 2) and (twoCount == 500):
                validValueCheck = True
            elif (whatValue == 3) and (threeCount == 500):
                validValueCheck = True
            elif (whatValue == 4) and (fourCount == 500):
                validValueCheck = True
            elif (whatValue == 5) and (fiveCount == 500):
                validValueCheck = True
            elif (whatValue == 6) and (sixCount == 500):
                validValueCheck = True
            elif (whatValue == 7) and (sevenCount == 500):
                validValueCheck = True
            elif (whatValue == 8) and (eightCount == 500):
                validValueCheck = True
            elif (whatValue == 9) and (nineCount == 500):
                validValueCheck = True
            else:
                validValueCheck = False
            if validValueCheck:
                whatValue = np.random.randint(10)

        # print(zeroOrOne)
        if whatValue == 0:
            if zeroIterator < 100:
                testLabels.append(whatValue)
                testDataSet.append(zeroSet[zeroCount])
            else:
                trainLabels.append(whatValue)
                trainDataSet.append(zeroSet[zeroCount])
            zeroIterator = zeroIterator + 1
            zeroCount = zeroCount + 1

        elif whatValue == 1:
            if oneIterator < 100:
                testLabels.append(whatValue)
                testDataSet.append(oneSet[oneCount])
            else:
                trainLabels.append(whatValue)
                trainDataSet.append(oneSet[oneCount])
            oneIterator = oneIterator + 1
            oneCount = oneCount + 1

        elif whatValue == 2:
            if twoIterator < 100:
                testLabels.append(whatValue)
                testDataSet.append(twoSet[twoCount])
            else:
                trainLabels.append(whatValue)
                trainDataSet.append(twoSet[twoCount])
            twoIterator = twoIterator + 1
            twoCount = twoCount + 1

        elif whatValue == 3:
            if threeIterator < 100:
                testLabels.append(whatValue)
                testDataSet.append(threeSet[threeCount])
            else:
                trainLabels.append(whatValue)
                trainDataSet.append(threeSet[threeCount])
            threeIterator = threeIterator + 1
            threeCount = threeCount + 1

        elif whatValue == 4:
            if fourIterator < 100:
                testLabels.append(whatValue)
                testDataSet.append(fourSet[fourCount])
            else:
                trainLabels.append(whatValue)
                trainDataSet.append(fourSet[fourCount])
            fourIterator = fourIterator + 1
            fourCount = fourCount + 1

        elif whatValue == 5:
            if fiveIterator < 100:
                testLabels.append(whatValue)
                testDataSet.append(fiveSet[fiveCount])
            else:
                trainLabels.append(whatValue)
                trainDataSet.append(fiveSet[fiveCount])
            fiveIterator = fiveIterator + 1
            fiveCount = fiveCount + 1

        elif whatValue == 6:
            if sixIterator < 100:
                testLabels.append(whatValue)
                testDataSet.append(sixSet[sixCount])
            else:
                trainLabels.append(whatValue)
                trainDataSet.append(sixSet[sixCount])
            sixIterator = sixIterator + 1
            sixCount = sixCount + 1

        elif whatValue == 7:
            if sevenIterator < 100:
                testLabels.append(whatValue)
                testDataSet.append(sevenSet[sevenCount])
            else:
                trainLabels.append(whatValue)
                trainDataSet.append(sevenSet[sevenCount])
            sevenIterator = sevenIterator + 1
            sevenCount = sevenCount + 1

        elif whatValue == 8:
            if eightIterator < 100:
                testLabels.append(whatValue)
                testDataSet.append(eightSet[eightCount])
            else:
                trainLabels.append(whatValue)
                trainDataSet.append(eightSet[eightCount])
            eightIterator = eightIterator + 1
            eightCount = eightCount + 1

        elif whatValue == 9:
            if nineIterator < 100:
                testLabels.append(whatValue)
                testDataSet.append(nineSet[nineCount])
            else:
                trainLabels.append(whatValue)
                trainDataSet.append(nineSet[nineCount])
            nineIterator = nineIterator + 1
            nineCount = nineCount + 1

        totalIterator = totalIterator + 1
    #print(totalIterator)
    np.savetxt('testDataSet.txt', testDataSet, fmt='%1.4f', delimiter=',')
    np.savetxt('trainDataSet.txt', trainDataSet, fmt='%1.4f', delimiter=',')
    np.savetxt('trainLabels.txt', trainLabels, fmt='%i', delimiter=',')
    np.savetxt('testLabels.txt', testLabels, fmt='%i', delimiter=',')

def InitialWeightCreation():
    inputWeights = np.random.uniform(-0.5, 0.5, size=[784,150])
    hiddenNeuronWeights = np.random.uniform(-0.5, 0.5, size=[150,10])
    np.savetxt('InitialWeights.txt', inputWeights, fmt='%1.4f', delimiter=',')
    np.savetxt('InitialHiddenWeights.txt', hiddenNeuronWeights, fmt='%1.4f', delimiter=',')
