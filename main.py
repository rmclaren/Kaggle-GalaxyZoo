
import os
import glob
from Solution import *
from SimpleCV import Image
import numpy as np
from GalaxyFactory import GalaxyFactory
import config

from Classifiers.Class1 import Class1
from Classifiers.Class2 import Class2
from Classifiers.Class3 import Class3
from Classifiers.Class4 import Class4
from Classifiers.Class5 import Class5
from Classifiers.Class6 import Class6
from Classifiers.Class7 import Class7
from Classifiers.Class8 import Class8
from Classifiers.Class9 import Class9
from Classifiers.Class10 import Class10
from Classifiers.Class11 import Class11


def run():
    config.initialize()

    #startPos = 5000
    #numSamples = 100
    #generateSolutionsFromTrainingSet(startPos, numSamples)
    #print "Result: %f" % scoreTrainingSolutions(startPos, numSamples)

    generateTestResults()


def generateSolutionsFromTrainingSet(startPos, numSamples):
    classifiers = getClassifiers()

    print "Classifying Solutions"
    galaxyFactory = GalaxyFactory()

    reader = SolutionReader(config.TRAINING_SOLUTIONS_PATH)
    results = SolutionWriter(config.TRAINING_SOLUTION_OUTPUT_PATH)
    reader.skipTo(startPos)

    sampleNum = 0
    solution = reader.next()
    while solution:
        imagePath = os.path.join(config.TRAINING_IMAGE_PATH, str(solution.GalaxyID) + '.jpg')
        galaxy = galaxyFactory.getGalaxyForImage(imagePath)

        predictedSolution = [galaxy.id]
        for classifier in classifiers:
            predictedSolution.extend(classifier.predict(galaxy))

        if len(solution) != 38:
            print "Problem with galaxy with id %s. Solution is not the correct length." % galaxy.id
        else:
            results.append(Solution._make(predictedSolution))

        sampleNum += 1
        if sampleNum > numSamples:
            break

        solution = reader.next()

    results.close()

def scoreTrainingSolutions(startPos, numSamples):
    return scoreResults(config.TRAINING_SOLUTIONS_PATH, config.TRAINING_SOLUTION_OUTPUT_PATH, startPos=startPos, numSamples=numSamples)

def generateTestResults():
    classifiers = getClassifiers()

    print "Scoring Solutions"
    galaxyFactory = GalaxyFactory()

    results = SolutionWriter(config.TEST_SOLUTION_OUTPUT_PATH)
    #numSamples =100
    sampleNum = 0
    for imagePath in glob.glob(config.TEST_IMAGE_PATH + '/*.jpg'):
        galaxy = galaxyFactory.getGalaxyForImage(imagePath)

        solution = [galaxy.id]
        for classifier in classifiers:
            predictions = classifier.predict(galaxy)
            predictions = [prediction if prediction >= 0.0 else 0.0 for prediction in predictions]
            solution.extend(predictions)

        if len(solution) != 38:
            print "Problem with galaxy with id %s. Solution is not the correct length." % galaxy.id
        else:
            results.append(Solution._make(solution))

        if not sampleNum % 10:
            print sampleNum

        sampleNum += 1
        #if sampleNum > numSamples:
        #    break

    results.close()

def getClassifiers():
    print 'Training Classifiers'

    classifiers = [Class1(), Class2(), Class3(), Class4(), Class5(), Class6(), Class7(), Class8(), Class9(), Class10(),
                   Class11()]

    for classifier in classifiers:
        classifier.load(getClassifierPath(classifier))

        if not classifier.areModelsTrained:
            classifier.train()
            classifier.save(getClassifierPath(classifier))

    return classifiers


def getClassifierPath(classifier):
    return os.path.join(config.CLASSIFIERS_PATH, classifier.__class__.__name__ + '.cls')

if __name__ == '__main__':
    run()
    #
    # config.initialize()
    # classifier = Class6()
    # classifier.plot()
    #
    # config.initialize()
    # classifier = Class1()
    # classifier.plot()
    #
    # config.initialize()
    # classifier3 = Class3()
    # classifier3.plot()
    #
    # config.initialize()
    # classifier4 = Class4()
    # classifier4.plot()
    #
    # config.initialize()
    # classifier5 = Class5()
    # classifier5.plot()
