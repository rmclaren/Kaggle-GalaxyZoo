from ClassBase import ClassBase
from sklearn.ensemble import RandomForestRegressor
from config import TRAINING_SOLUTIONS_PATH, TRAINING_IMAGE_PATH

from Solution import *
from GalaxyFactory import GalaxyFactory
import matplotlib.pyplot as plt

class Class9(ClassBase):

    def __init__(self):
        super(Class9, self).__init__()
        self.ellipticalModel = None
        self.rectangularModel = None
        self.flatModel = None

    def predict(self, galaxy):
        features = self._extractFeatures(galaxy)
        if not features or galaxy.ellipse.ellipse_area() == 0:
            return self._defaultPrediction()

        featureArray = np.array([features['bulgeIntensity'],
                                 features['bulgeBlobFractionalArea'],
                                 features['fractionForeground']], dtype=np.float32)

        ellipticalResult = self.ellipticalModel.predict(featureArray)[0]
        rectangularResult = self.rectangularModel.predict(featureArray)[0]
        flatResult = self.flatModel.predict(featureArray)[0]

        return [ellipticalResult, rectangularResult, flatResult]

    def _defaultPrediction(self):
        return [0.5921896703061242, 0.0997483267204096, 0.30806200297346625]

    def train(self, verbose=False, training_data=None):
        if not training_data:
            trainingDataDict = self._getTrainingData(numSamples=5000)
        else:
            trainingDataDict = training_data

        X = np.array(trainingDataDict['training_data'], dtype=np.float32)

        y = np.array(trainingDataDict['solution_data'][0], dtype=np.float32)
        self.ellipticalModel = RandomForestRegressor(max_depth=50)
        self.ellipticalModel = self.ellipticalModel.fit(X, y)

        y = np.array(trainingDataDict['solution_data'][1], dtype=np.float32)
        self.rectangularModel = RandomForestRegressor(max_depth=50)
        self.rectangularModel = self.rectangularModel.fit(X, y)

        y = np.array(trainingDataDict['solution_data'][2], dtype=np.float32)
        self.flatModel = RandomForestRegressor(max_depth=50)
        self.flatModel = self.flatModel.fit(X, y)

        self.areModelsTrained = True

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
                if galaxy.ellipse.ellipse_area() > 0:
                    features = self._extractFeatures(galaxy, useCache=True)

                    if features:
                        training_data.append([features['bulgeIntensity'],
                                              features['bulgeBlobFractionalArea'],
                                              features['fractionForeground']])

                        solution_data[0].append(trainingSol.Class9_1)
                        solution_data[1].append(trainingSol.Class9_2)
                        solution_data[2].append(trainingSol.Class9_3)

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

        return {'galaxyIds': galaxyIds, 'training_data': training_data, 'solution_data': solution_data}

    def plot(self):
        print 'Plotting Class 9'

        trainingData = self._getTrainingData(numSamples=1000)

        if not self.areModelsTrained:
            self.train(verbose=True, training_data=trainingData)

        trainingData = self._getTrainingData(startPos=1000, numSamples=1000)

        plotRange = range(len(trainingData['training_data']))
        model1Results = [self.ellipticalModel.predict(np.array(trainingData['training_data'][i], dtype=np.float32)) for i in plotRange]
        model1Results = [model1Results[i][0] for i in xrange(len(model1Results))]
        plt.plot(trainingData['solution_data'][0], model1Results, 'r+')

        model2Results = [self.rectangularModel.predict(np.array(trainingData['training_data'][i], dtype=np.float32)) for i in plotRange]
        model2Results = [model2Results[i][0] for i in xrange(len(model2Results))]
        plt.plot(trainingData['solution_data'][1], model2Results, 'g+')

        model3Results = [self.flatModel.predict(np.array(trainingData['training_data'][i], dtype=np.float32)) for i in plotRange]
        model3Results = [model3Results[i][0] for i in xrange(len(model3Results))]
        plt.plot(trainingData['solution_data'][2], model3Results, 'b+')

        plt.title('Class 9')

        print '    Showing Class 9'

        plt.show()