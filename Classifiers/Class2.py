from ClassBase import ClassBase
from GalaxyFactory import GalaxyFactory
from config import TRAINING_SOLUTIONS_PATH, TRAINING_IMAGE_PATH
from Solution import *
from sklearn.svm import SVR
import numpy as np
import math

class Class2(ClassBase):

    def __init__(self):
        super(Class2, self).__init__()
        self.flatGalaxyModel = None
        self.nonFlatGalaxyModel = None

    def predict(self, galaxy):
        featureArray = np.array([galaxy.getAspectRatio()], dtype=np.float32)

        flatResult = self.flatGalaxyModel.predict(featureArray)[0]
        #nonFlatResult = self.nonFlatGalaxyModel.predict(featureArray)[0]

        return [flatResult, 1-flatResult]

    def train(self, verbose=False, training_data=None):
        trainingDataDict = self._getTrainingData(numSamples=5000)

        X = np.array(trainingDataDict['training_data'], dtype=np.float32)

        y = np.array(trainingDataDict['solution_data'][0], dtype=np.float32)
        self.flatGalaxyModel = SVR(C=50)
        self.flatGalaxyModel = self.flatGalaxyModel.fit(X, y)

        y = np.array(trainingDataDict['solution_data'][1], dtype=np.float32)
        self.nonFlatGalaxyModel = SVR(C=50)
        self.nonFlatGalaxyModel = self.nonFlatGalaxyModel.fit(X, y)

        self.areModelsTrained = True

    def _defaultPrediction(self):
        return [0.19622986674338236, 0.8037701332566176]

    def _getTrainingData(self, startPos=0, numSamples=5000):
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

                solution_data[0].append(trainingSol.Class2_1)
                solution_data[1].append(trainingSol.Class2_2)

            galaxyIndex += 1
            if galaxyIndex > numSamples:
                break

            trainingSol = sr.next()

        return {'galaxyIds': galaxyIds, 'training_data': training_data, 'solution_data': solution_data}
