from ClassBase import ClassBase

import matplotlib.pyplot as plt

from config import TRAINING_SOLUTIONS_PATH, TRAINING_IMAGE_PATH
from Solution import *
from GalaxyFactory import GalaxyFactory

from sklearn.ensemble import RandomForestRegressor

class Class3(ClassBase):

    # def __init__(self):
    #     super(Class3, self).__init__()
    #     self.barGalaxyModel = None
    #     self.training_data = None
    #
    # def predict(self, galaxy):
    #     if galaxy.rotLineTestDerivative != None:
    #         featureArray = np.array([galaxy.rotBoxTestResult], dtype=np.float32)
    #         barResult = self.barGalaxyModel.predict(featureArray)[0]
    #
    #         return [barResult, 1-barResult]
    #
    #     return self._defaultPrediction()
    #
    # def train(self, verbose=False, training_data=None):
    #     if not training_data:
    #         trainingDataDict = self._getTrainingData(numSamples=5000)
    #     else:
    #         trainingDataDict = training_data
    #
    #     X = np.array(trainingDataDict['rotateBoxTestResult'], dtype=np.float32)
    #     y = np.array(trainingDataDict['solution_data'][0], dtype=np.float32)
    #
    #     self.barGalaxyModel = RandomForestRegressor(max_depth=50, max_features="log2")
    #     self.barGalaxyModel = self.barGalaxyModel.fit(X, y)
    #
    #     self.areModelsTrained = True

    def _defaultPrediction(self):
        return [0.23523014587453597, 0.764769854125464]

    # def _getTrainingData(self, startPos=0, numSamples=5000, verbose=False):
    #     galaxyFactory = GalaxyFactory(isTrainingData=True)
    #     sr = SolutionReader(TRAINING_SOLUTIONS_PATH)
    #     sr.skipTo(startPos)
    #
    #     galaxyIds = []
    #     solution_data = [[], []]
    #
    #     rotateLineTest = []
    #     rotateBoxResult = []
    #     scaledImg = []
    #
    #     galaxyIndex = 0
    #
    #     trainingSol = sr.next()
    #     while trainingSol:
    #         galaxy = galaxyFactory.getGalaxyForImage(os.path.join(TRAINING_IMAGE_PATH,
    #                                                               str(trainingSol.GalaxyID) + '.jpg'))
    #
    #         features = self._extractFeatures(galaxy, useCache=True)
    #
    #         if features != None and features['rotLineTestResult'] != None and features['scaledImg'] != None:
    #             galaxyIds.append(galaxy.id)
    #             rotateLineTest.append(features['rotLineTestResult'])
    #             rotateBoxResult.append(features['rotateBoxTestResult'])
    #             scaledImg.append(features['scaledImg'])
    #
    #             solution_data[0].append(trainingSol.Class3_1)
    #             solution_data[1].append(trainingSol.Class3_2)
    #
    #         if not galaxyIndex%10:
    #             print galaxyIndex
    #
    #         galaxyIndex += 1
    #         if galaxyIndex > numSamples:
    #             break
    #
    #         trainingSol = sr.next()
    #
    #     return {'galaxyIds': galaxyIds,
    #             'rotateLineTest': rotateLineTest,
    #             'rotateBoxTestResult': rotateBoxResult,
    #             'scaledImg': scaledImg,
    #             'solution_data': solution_data}
    #
    # def plot(self):
    #     print 'Plotting Class 3'
    #
    #     trainingData = self._getTrainingData(numSamples=500, verbose=True)
    #
    #     if not self.areModelsTrained:
    #         self.train(verbose=True, training_data=trainingData)
    #
    #     trainingData = self._getTrainingData(startPos=5000, numSamples=500, verbose=True)
    #
    #     plotRange = range(len(trainingData['rotateBoxTestResult']))
    #     barResults = [self.barGalaxyModel.predict(np.array(trainingData['rotateBoxTestResult'][i], dtype=np.float32)) for i in plotRange]
    #     #barResults1 = [barResults[i][0][0] for i in xrange(len(barResults))]
    #     #barResults2 = [barResults[i][0][1] for i in xrange(len(barResults))]
    #     plt.plot(trainingData['solution_data'][0], barResults, 'r+')
    #     #plt.plot(trainingData['solution_data'][1], barResults2, 'b+')
    #
    #     plt.title('Class 3')
    #
    #     print '    Showing Class 3'
    #
    #     plt.show()

