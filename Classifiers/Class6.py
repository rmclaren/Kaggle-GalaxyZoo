from ClassBase import ClassBase
from sklearn.ensemble import RandomForestRegressor
from config import TRAINING_SOLUTIONS_PATH, TRAINING_IMAGE_PATH

from Solution import *
from GalaxyFactory import GalaxyFactory
import matplotlib.pyplot as plt

class Class6(ClassBase):
    #
    # def __init__(self):
    #     super(Class6, self).__init__()
    #     self.somethingWeirdModel = None
    #
    # def predict(self, galaxy):
    #     features = self._extractFeatures(galaxy)
    #     if not features or galaxy.ellipse.ellipse_area() == 0:
    #         return self._defaultPrediction()
    #
    #     featureArray = np.array([features['fractionForeground'],
    #                              galaxy._findGalaxyBulgeBlobs()], dtype=np.float32)
    #
    #     result = self.somethingWeirdModel.predict(featureArray)[0]
    #
    #     return [result, 1-result]

    def _defaultPrediction(self):
        return [0.231807, 0.768193]

    # def train(self, verbose=False, training_data=None):
    #     if not training_data:
    #         trainingDataDict = self._getTrainingData(numSamples=5000)
    #     else:
    #         trainingDataDict = training_data
    #
    #     X = np.array(trainingDataDict['training_data'], dtype=np.float32)
    #
    #     y = np.array(trainingDataDict['solution_data'][0], dtype=np.float32)
    #     self.somethingWeirdModel = RandomForestRegressor(max_depth=50)
    #     self.somethingWeirdModel = self.somethingWeirdModel.fit(X, y)
    #
    #     self.areModelsTrained = True
    #
    # def _getTrainingData(self, startPos=0, numSamples=5000, verbose=False):
    #     galaxyFactory = GalaxyFactory(isTrainingData=True)
    #     sr = SolutionReader(TRAINING_SOLUTIONS_PATH)
    #     sr.skipTo(startPos)
    #
    #     galaxyIds = []
    #     solution_data = [[], [], []]
    #     training_data = []
    #     success = []
    #
    #     galaxyIndex = 0
    #
    #     trainingSol = sr.next()
    #     while trainingSol:
    #         galaxy = galaxyFactory.getGalaxyForImage(os.path.join(TRAINING_IMAGE_PATH,
    #                                                               str(trainingSol.GalaxyID) + '.jpg'))
    #
    #         if galaxy.initSuccess:
    #             if galaxy.ellipse.ellipse_area() > 0:
    #                 features = self._extractFeatures(galaxy, useCache=True)
    #
    #                 if features:
    #                     training_data.append([features['fractionForeground'],
    #                                           len(galaxy._findGalaxyBulgeBlobs()])
    #
    #                     solution_data[0].append(trainingSol.Class6_1)
    #                     solution_data[1].append(trainingSol.Class6_2)
    #
    #                     galaxyIds.append(trainingSol.GalaxyID)
    #                     success.append(True)
    #
    #                     if verbose and not galaxyIndex % 10:
    #                         print galaxyIndex
    #
    #                     galaxyIndex += 1
    #                     if galaxyIndex > numSamples:
    #                         break
    #
    #         else:
    #             galaxyIds.append(trainingSol.GalaxyID)
    #             success.append(False)
    #
    #         trainingSol = sr.next()
    #
    #     return {'galaxyIds': galaxyIds, 'training_data': training_data, 'solution_data': solution_data}
    #
    # def plot(self):
    #     print 'Plotting Class 6'
    #
    #     trainingData = self._getTrainingData(numSamples=10)
    #
    #     if not self.areModelsTrained:
    #         self.train(verbose=True, training_data=trainingData)
    #
    #     trainingData = self._getTrainingData(startPos=1000, numSamples=10)
    #
    #     plotRange = range(len(trainingData['training_data']))
    #     results = [self.somethingWeirdModel.predict(np.array(trainingData['training_data'][i], dtype=np.float32)) for i in plotRange]
    #     results = [results[i][0] for i in xrange(len(results))]
    #     plt.plot(trainingData['solution_data'][0], results, 'r+')
    #
    #     plt.title('Class 6')
    #
    #     print '    Showing Class 6'
    #
    #     plt.show()