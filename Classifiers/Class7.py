from ClassBase import ClassBase
from GalaxyFactory import GalaxyFactory
from config import TRAINING_SOLUTIONS_PATH, TRAINING_IMAGE_PATH
from Solution import *
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np
import math

class Class7(ClassBase):

    def __init__(self):
        super(Class7, self).__init__()
        self.roundGalaxyModel = None
        self.ovalGalaxyModel = None
        self.cigarShapedGalaxyModel = None

    def predict(self, galaxy):
        featureArray = np.array([galaxy.getAspectRatio()], dtype=np.float32)

        roundResult = self.roundGalaxyModel.predict(featureArray)[0]
        ovalResult = self.ovalGalaxyModel.predict(featureArray)[0]
        cigarResult = self.cigarShapedGalaxyModel.predict(featureArray)[0]

        return [roundResult, 1-roundResult-cigarResult, cigarResult]

    def train(self, verbose=False, training_data=None):
        if not training_data:
            trainingDataDict = self._getTrainingData(numSamples=5000, verbose=verbose)
        else:
            trainingDataDict = training_data

        X = np.array(trainingDataDict['training_data'], dtype=np.float32)

        y = np.array(trainingDataDict['solution_data'][0], dtype=np.float32)
        self.roundGalaxyModel = SVR(C=50)
        self.roundGalaxyModel = self.roundGalaxyModel.fit(X, y)

        y = np.array(trainingDataDict['solution_data'][1], dtype=np.float32)
        self.ovalGalaxyModel = SVR(C=50)
        self.ovalGalaxyModel = self.ovalGalaxyModel.fit(X, y)

        y = np.array(trainingDataDict['solution_data'][2], dtype=np.float32)
        self.cigarShapedGalaxyModel = SVR(C=50)
        self.cigarShapedGalaxyModel = self.cigarShapedGalaxyModel.fit(X, y)

        self.areModelsTrained = True

    def _defaultPrediction(self):
        return [0.393088598856161, 0.4803527372580463, 0.12655866388579273]

    def _getTrainingData(self, startPos=0, numSamples=5000, verbose=False):
        galaxyFactory = GalaxyFactory(isTrainingData=True)
        sr = SolutionReader(TRAINING_SOLUTIONS_PATH)
        sr.skipTo(startPos)

        galaxyIds = []
        solution_data = [[], [], []]
        training_data = []

        galaxyIndex = 0

        trainingSol = sr.next()
        while trainingSol:
            galaxy = galaxyFactory.getGalaxyForImage(os.path.join(TRAINING_IMAGE_PATH,
                                                                  str(trainingSol.GalaxyID) + '.jpg'))

            aspectRatio = galaxy.getAspectRatio()

            if not math.isnan(aspectRatio):
                galaxyIds.append(galaxy.id)

                training_data.append([aspectRatio])

                solution_data[0].append(trainingSol.Class7_1)
                solution_data[1].append(trainingSol.Class7_2)
                solution_data[2].append(trainingSol.Class7_3)

            galaxyIndex += 1
            if galaxyIndex > numSamples:
                break

            if verbose and not galaxyIndex % 10:
                print galaxyIndex

            trainingSol = sr.next()

        return {'galaxyIds': galaxyIds, 'training_data': training_data, 'solution_data': solution_data}

    def load(self, path):
        if (os.path.exists(path)):
            classifier = pickle.load(open(path, 'rb'))
            self.roundGalaxyModel = classifier.roundGalaxyModel
            self.ovalGalaxyModel = classifier.ovalGalaxyModel
            self.cigarShapedGalaxyModel = classifier.cigarShapedGalaxyModel

            self.areModelsTrained = classifier.areModelsTrained

    def plot(self):
        trainingData = self._getTrainingData(numSamples=500, verbose=True)

        if not self.areModelsTrained:
            self.train(verbose=True, training_data=trainingData)

        index = 0
        plt.plot(trainingData['training_data'], np.array(trainingData['solution_data'][index]), 'r+', alpha=0.5)

        index = 1
        plt.plot(trainingData['training_data'], np.array(trainingData['solution_data'][index]), 'g+', alpha=0.5)

        index = 2
        plt.plot(trainingData['training_data'], np.array(trainingData['solution_data'][index]), 'b+', alpha=0.5)

        plotRange = np.arange(0, 1, .01)
        plt.plot(plotRange,
                 [self.roundGalaxyModel.predict(np.array(i, dtype=np.float32)) for i in plotRange], 'r')

        plt.plot(plotRange,
                 [self.ovalGalaxyModel.predict(np.array(i, dtype=np.float32)) for i in plotRange], 'g')

        plt.plot(plotRange,
                 [self.cigarShapedGalaxyModel.predict(np.array(i, dtype=np.float32)) for i in plotRange], 'b')

        plt.show()

if __name__ == '__main__':
    classifier = Class7()
    classifier.plot()
