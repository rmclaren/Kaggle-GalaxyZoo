#This file encapsulates all the knowledge we have of the Solution files.
#It knows how to read and write these files given the raw data.

import os
import numpy as np
import math
import csv
import pickle
from collections import namedtuple

_solutionFields = ['GalaxyID', 'Class1_1', 'Class1_2', 'Class1_3', 'Class2_1', 'Class2_2', 'Class3_1',
                   'Class3_2', 'Class4_1', 'Class4_2', 'Class5_1', 'Class5_2', 'Class5_3', 'Class5_4',
                   'Class6_1', 'Class6_2', 'Class7_1', 'Class7_2', 'Class7_3', 'Class8_1', 'Class8_2',
                   'Class8_3', 'Class8_4', 'Class8_5', 'Class8_6', 'Class8_7', 'Class9_1', 'Class9_2',
                   'Class9_3', 'Class10_1', 'Class10_2', 'Class10_3', 'Class11_1', 'Class11_2',
                   'Class11_3', 'Class11_4', 'Class11_5', 'Class11_6']

Solution = namedtuple('Solution', _solutionFields)

HEADER = 'GalaxyID,Class1.1,Class1.2,Class1.3,Class2.1,Class2.2,Class3.1,Class3.2,Class4.1,Class4.2,Class5.1,Class5.2,\
Class5.3,Class5.4,Class6.1,Class6.2,Class7.1,Class7.2,Class7.3,Class8.1,Class8.2,Class8.3,Class8.4,Class8.5,Class8.6,\
Class8.7,Class9.1,Class9.2,Class9.3,Class10.1,Class10.2,Class10.3,Class11.1,Class11.2,Class11.3,Class11.4,Class11.5,\
Class11.6'


class SolutionReader(object):
    def __init__(self, inputPath):
        self.csvFile = csv.reader(open(inputPath, "rb"))
        self.solutionIter = iter(map(Solution._make, self.csvFile))

        #Throw away the first line
        self.solutionIter.next()

        self.solutionsLeft = self.solutionIter.__length_hint__()
        self.numberOfTargets = self.solutionsLeft

    def next(self):
        result = None
        if self.solutionsLeft > 0:
            self.solutionsLeft -= 1
            result = [float(val) for val in self.solutionIter.next()[:]]
            result[0] = int(result[0])
            result = self._normalizeSolution(result)
            result = Solution._make(result)

        return result

    def nextUnscaled(self):
        result = None
        if self.solutionsLeft > 0:
            self.solutionsLeft -= 1
            result = [float(val) for val in self.solutionIter.next()[:]]
            result[0] = int(result[0])
            result = Solution._make(result)

        return result


    def skipTo(self, pos):
        for i in xrange(pos):
            self.next()

    def _normalizeSolution(self, sol):
        fields = _solutionFields[1:]
        classes = sol[1:]
        result = [sol[0]]
        classList = []

        for i, field in enumerate(fields):
            info = field.split('_')
            index = int(info[0].split('Class')[1]) - 1
            if index > len(classList) - 1:
                classList.append([])
            classList[index].append(classes[i])

        for values in classList:
            s = sum(values)
            if s == 0:
                C = 1.0
            else:
                C = 1/s

            result.extend([C*v for v in values])

        return result


class SolutionWriter(object):
    def __init__(self, outputPath):
        self.outputFile = open(outputPath, 'w')
        self.outputFile.write(HEADER + '\n')
        self.decisionTree = getDecisionTree()

    def append(self, solution):
        scaledSolution = dict.fromkeys(_solutionFields, 0)
        for key in scaledSolution:
            scaledSolution[key] = getattr(solution, key)

        SolutionWriter._scaleSolution(scaledSolution)
        self.outputFile.write(','.join([str(scaledSolution[key]) for key in _solutionFields]) + '\r\n')

    def close(self):
        self.outputFile.close()

    @staticmethod
    def _scaleSolution(scaledSolution):
        scaleValue = scaledSolution['Class1_1']
        for key in ['Class7_1', 'Class7_2', 'Class7_3']:
            scaledSolution[key] *= scaleValue

        scaleValue = scaledSolution['Class1_2']
        for key in ['Class2_1', 'Class2_2']:
            scaledSolution[key] *= scaleValue

        scaleValue = scaledSolution['Class2_1']
        for key in ['Class9_1', 'Class9_2', 'Class9_3']:
            scaledSolution[key] *= scaleValue

        scaleValue = scaledSolution['Class2_2']
        for key in ['Class3_1', 'Class3_2']:
            scaledSolution[key] *= scaleValue

        scaleValue = scaledSolution['Class2_2']
        for key in ['Class4_1', 'Class4_2']:
            scaledSolution[key] *= scaleValue

        scaleValue = scaledSolution['Class4_1']
        for key in ['Class10_1', 'Class10_2', 'Class10_3']:
            scaledSolution[key] *= scaleValue

        scaleValue = scaledSolution['Class4_1']
        for key in ['Class11_1', 'Class11_2', 'Class11_3', 'Class11_4', 'Class11_5', 'Class11_6']:
            scaledSolution[key] *= scaleValue

        scaleValue = scaledSolution['Class4_1'] + scaledSolution['Class4_2']
        for key in ['Class5_1', 'Class5_2', 'Class5_3', 'Class5_4']:
            scaledSolution[key] *= scaleValue

        scaleValue = scaledSolution['Class6_1']
        for key in ['Class8_1', 'Class8_2', 'Class8_3', 'Class8_4', 'Class8_5', 'Class8_6', 'Class8_7']:
            scaledSolution[key] *= scaleValue


def scoreResults(trainingFilePath, myResultFilePath, startPos=0, numSamples=0):
    print "Scoring Results"
    myResultSR = SolutionReader(myResultFilePath)
    trainingResultSR = SolutionReader(trainingFilePath)
    trainingResultSR.skipTo(startPos)

    numberOfClasses = len(_solutionFields) - 1

    if numSamples:
        N = float(min(numSamples, myResultSR.solutionsLeft) * numberOfClasses)
    else:
        N = float(myResultSR.solutionsLeft * numberOfClasses)

    count = 0
    runningSum = 0
    mySol = myResultSR.nextUnscaled()
    trainSol = trainingResultSR.nextUnscaled()
    while mySol:
        if mySol.GalaxyID != trainSol.GalaxyID:
            print "ERROR: GalaxyId did not match %s %s" % (mySol.GalaxyID, trainSol.GalaxyID)
            break

        #runningSum += sum([(mySol[i] - trainSol[i])**2 for i in xrange(1, 38)])

        runningSum += np.sum((np.array(mySol[-1*numberOfClasses:]) - np.array(trainSol[-1*numberOfClasses:]))**2)

        if not count % 1000:
            print count

        mySol = myResultSR.next()
        trainSol = trainingResultSR.next()
        count += 1

        if numSamples and count >= numSamples:
            break

    return math.sqrt(1/N * runningSum)

#Misc utility functions

def makeTrainFile(trainingFilePath):
    trainingResultSR = SolutionReader(trainingFilePath)
    sol = trainingResultSR.next()

    zerosSW = SolutionWriter(os.path.realpath('./training_solutions/ones.csv'))

    while (sol):
        newSolVals = [1] * len(_solutionFields)
        newSolVals[0] = sol[0]
        zerosSW.append(Solution._make(newSolVals))
        sol = trainingResultSR.next()

    zerosSW.close()

def makeAverageSolutionFile(trainingFilePath):
    trainingResultSR = SolutionReader(trainingFilePath)

    averageSW = SolutionWriter(os.path.realpath('./training_solutions/average.csv'))

    averageArray = np.array([0] * (len(_solutionFields) - 1))

    sol = trainingResultSR.next()
    while (sol):
        averageArray = averageArray + sol[-1*(len(_solutionFields)-1):]
        sol = trainingResultSR.next()

    averageArray = (1.0/trainingResultSR.numberOfTargets) * averageArray
    averageArray = np.array([0] + averageArray.tolist())

    trainingResultSR = SolutionReader(trainingFilePath)

    sol = trainingResultSR.next()
    while (sol):
        averageArray[0] = sol.GalaxyID
        averageSW.append(Solution._make(averageArray))
        sol = trainingResultSR.next()

    averageSW.close()

def detectIdentImages():
    import glob

    trainingTitleDict = {}
    for file in glob.glob('./images_training/*.jpg'):
        trainingTitleDict[os.path.split(file)[1]] = None

    for file in glob.glob('./images_test/*.jpg'):
        if os.path.split(file)[1] in trainingTitleDict:
            print 'MATCH FOUND!'

def getDecisionTree():
    class08 = {'Class8_1': None, 'Class8_2': None, 'Class8_3': None, 'Class8_4': None, 'Class8_5': None, 'Class8_6': None, 'Class8_7': None}
    class06 = {'Class6_1': class08, 'Class6_2': None}
    class05 = {'Class5_1': class06, 'Class5_2': class06, 'Class5_3': class06, 'Class5_4': class06}
    class07 = {'Class7_1': class06, 'Class7_2': class06, 'Class7_3': class06}
    class09 = {'Class9_1': class06, 'Class9_2': class06, 'Class9_3': class06}
    class11 = {'Class11_1': class05, 'Class11_2': class05, 'Class11_3': class05, 'Class11_4': class05, 'Class11_5': class05, 'Class11_6': class05}
    class10 = {'Class10_1': class11, 'Class10_2': class11, 'Class10_3': class11}
    class04 = {'Class4_1': class10, 'Class4_2': class05}
    class03 = {'Class3_1': class04, 'Class3_2': class04}
    class02 = {'Class2_1': class09, 'Class2_2': class03}
    class01 = {'Class1_1': class07, 'Class1_2': class02, 'Class1_3': None}

    return class01

def createPositivesFile(input, output, posThreshold=0.75, negThreshold=0.15):
    decisionTree = getDecisionTree()
    sr = SolutionReader(input)
    maxClasses = []
    sol = sr.next()
    while sol:
        result = []
        _findPositives(sol, decisionTree, result)

        solContents = [0]*len(sol)
        solContents[0] = sol.GalaxyID
        for c in result:
            i = _solutionFields.index(c)
            if sol[i] >= posThreshold:
                solContents[i] = 1.0

        for i in xrange(1, len(sol)):
            if solContents[i] != 1.0 and sol[i] <= negThreshold:
                solContents[i] = -1.0

        maxClasses.append(tuple(solContents))

        sol = sr.next()

    pickle.dump(maxClasses, open(output, 'wb'))


def _findPositives(sol, decisionTree, result):
    maxKey = None
    maxValue = 0

    for key in decisionTree.keys():
        if getattr(sol, key) > maxValue:
            maxValue = getattr(sol, key)
            maxKey = key

    result.append(maxKey)

    if decisionTree[maxKey]:
        _findPositives(sol, decisionTree[maxKey], result)


if __name__ == '__main__':
    #sr = SolutionReader(os.path.realpath('./training_solutions/average.csv'))
    #sw = SolutionWriter(os.path.realpath('./training_solutions/test.csv'))

    #sol = sr.next()
    #sw.append(sol)
    #sw.close()

    #srt = SolutionReader(os.path.realpath('./training_solutions/test.csv'))
    #testSol = SolutionReader(os.path.realpath('./training_solutions/test.csv')).nextUnscaled()
    #testSol2 = SolutionReader(os.path.realpath('./training_solutions/average.csv')).nextUnscaled()

    #if sol == testSol:
    #    print 'OK'

    #createPositivesFile(os.path.realpath('./training_solutions/training_solutions_rev1.csv'),
    #                    os.path.realpath('./data/positives.dat'))

    #sr = SolutionReader(os.path.realpath('./training_solutions/training_solutions_rev1.csv'))
    #a = sr.next()
    #print a

    #makeAverageSolutionFile(os.path.realpath('./training_solutions/training_solutions_rev1.csv'))

    print '%s' % scoreResults(os.path.realpath('./training_solutions/training_solutions_rev1.csv'),
                              os.path.realpath('./training_solutions/zeros.csv'))
