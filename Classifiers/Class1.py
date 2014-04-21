import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from ClassBase import ClassBase
from config import TRAINING_SOLUTIONS_PATH, TRAINING_IMAGE_PATH
from Solution import *
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from GalaxyFactory import GalaxyFactory

class Class1(ClassBase):
    '''
        Identify the difference between galaxies that are smooth, have a disk, or are not galaxies at all.
    '''
    def __init__(self):
        super(Class1, self).__init__()

        self.smoothGalaxyModel = None
        self.diskGalaxyModel = None
        self.nonGalaxyModel = None

    def predict(self, galaxy):
        features = self._extractFeatures(galaxy)
        if not features:
            return self._defaultPrediction()

        featureArray = np.array([features['bulgeIntensity'],
                                 features['R'] - features['G'],
                                 features['asymScore'],
                                 abs(features['rotBias'])], dtype=np.float32)

        smoothResult = self.smoothGalaxyModel.predict(featureArray)[0]
        diskResult = self.diskGalaxyModel.predict(featureArray)[0]
        nonGalaxyResult = self.nonGalaxyModel.predict(featureArray)[0]

        return [smoothResult, diskResult, nonGalaxyResult]

    def _defaultPrediction(self):
        return [0.432526, 0.542261, 0.0252128]

    def train(self, verbose=False, training_data=None):
        if not training_data:
            trainingDataDict = self._getTrainingData(numSamples=5000, verbose=verbose)
        else:
            trainingDataDict = training_data

        X = np.array(trainingDataDict['training_data'], dtype=np.float32)

        y = np.array(trainingDataDict['solution_data'][0], dtype=np.float32)
        #self.smoothGalaxyModel = SVR(C=10)
        self.smoothGalaxyModel = RandomForestRegressor(max_depth=50)
        self.smoothGalaxyModel = self.smoothGalaxyModel.fit(X, y)

        y = np.array(trainingDataDict['solution_data'][1], dtype=np.float32)
        #self.diskGalaxyModel = SVR(C=10)
        self.diskGalaxyModel = RandomForestRegressor(max_depth=50)
        self.diskGalaxyModel = self.diskGalaxyModel.fit(X, y)

        y = np.array(trainingDataDict['solution_data'][2], dtype=np.float32)
        #self.nonGalaxyModel = SVR(C=10)
        self.nonGalaxyModel = RandomForestRegressor(max_depth=50)
        self.nonGalaxyModel = self.nonGalaxyModel.fit(X, y)

        self.areModelsTrained = True

    def load(self, path):
        if (os.path.exists(path)):
            class1classifier = pickle.load(open(path, 'rb'))
            self.smoothGalaxyModel = class1classifier.smoothGalaxyModel
            self.diskGalaxyModel = class1classifier.diskGalaxyModel
            self.nonGalaxyModel = class1classifier.nonGalaxyModel
            self.areModelsTrained = class1classifier.areModelsTrained

    def _getTrainingData(self, startPos=0, numSamples=5000, verbose=False):
        galaxyFactory = GalaxyFactory(isTrainingData=True)
        sr = SolutionReader(TRAINING_SOLUTIONS_PATH)
        sr.skipTo(startPos)

        galaxyIds = []
        solution_data = [[], [], []]
        training_data = []
        success = []

        galaxyIndex = 0

        trainingSol = sr.next()
        while trainingSol:
            galaxy = galaxyFactory.getGalaxyForImage(os.path.join(TRAINING_IMAGE_PATH,
                                                                  str(trainingSol.GalaxyID) + '.jpg'))

            if galaxy.initSuccess:
                features = self._extractFeatures(galaxy, useCache=True)

                if features:
                    training_data.append([features['bulgeIntensity'],
                                          features['R'] - features['G'],
                                          features['asymScore'],
                                          features['rotBias']])

                    solution_data[0].append(trainingSol.Class1_1)
                    solution_data[1].append(trainingSol.Class1_2)
                    solution_data[2].append(trainingSol.Class1_3)

                    galaxyIds.append(trainingSol.GalaxyID)
                    success.append(True)

                    if verbose and not galaxyIndex % 10:
                        print galaxyIndex

                    galaxyIndex += 1
                    if galaxyIndex > numSamples:
                        break

            else:
                galaxyIds.append(trainingSol.GalaxyID)
                success.append(False)

            trainingSol = sr.next()

        return {'galaxyIds': galaxyIds, 'training_data': training_data, 'solution_data': solution_data, 'success': success}


    def plot(self):
        trainingData = self._getTrainingData(numSamples=1000, verbose=True)

        if not self.areModelsTrained:
            self.train(verbose=True, training_data=trainingData)

        trainingData = self._getTrainingData(startPos=1000, numSamples=1000, verbose=True)

        plotRange = range(len(trainingData['training_data']))
        smoothResults = [self.smoothGalaxyModel.predict(np.array(trainingData['training_data'][i], dtype=np.float32)) for i in plotRange]
        smoothResults = [smoothResults[i][0] for i in xrange(len(smoothResults))]
        plt.plot(trainingData['solution_data'][0], smoothResults, 'r+')

        diskResults = [self.diskGalaxyModel.predict(np.array(trainingData['training_data'][i], dtype=np.float32)) for i in plotRange]
        diskResults = [diskResults[i][0] for i in xrange(len(diskResults))]
        plt.plot(trainingData['solution_data'][1], diskResults, 'g+')

        nonGalaxyResult = [self.nonGalaxyModel.predict(np.array(trainingData['training_data'][i], dtype=np.float32)) for i in plotRange]
        nonGalaxyResult = [nonGalaxyResult[i][0] for i in xrange(len(nonGalaxyResult))]
        plt.plot(trainingData['solution_data'][2], nonGalaxyResult, 'b+')

        plt.title('Class 1')

        print '    Showing Class 1'

        plt.show()


