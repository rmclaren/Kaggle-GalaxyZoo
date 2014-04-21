import os

runVersion = '0.3'
description = ''

#Unchanging data paths
TEST_IMAGE_PATH = os.path.realpath('./images_test/')
TRAINING_IMAGE_PATH = os.path.realpath('./images_training/')
TRAINING_SOLUTIONS_PATH = os.path.realpath('./training_solutions/training_solutions_rev1.csv')

#Data paths which will change with runVersion
runPath = os.path.join(os.path.realpath('./'), runVersion)
RUN_DESCRIPTION = os.path.join(runPath, 'description.txt')
TEST_GALAXY_PATH = os.path.join(runPath, 'test_galaxies/')
TRAINING_GALAXY_PATH = os.path.join(runPath, 'training_galaxies/')
CLASSIFIERS_PATH = os.path.join(runPath, 'classifiers/')
RESULT_SOLUTION_DIR = os.path.join(runPath, 'result_solutions/')
TEST_SOLUTION_OUTPUT_PATH = os.path.join(RESULT_SOLUTION_DIR, 'solution.csv')
TRAINING_SOLUTION_OUTPUT_PATH = os.path.join(RESULT_SOLUTION_DIR, 'training_solution.csv')

def initialize():
    changingPaths = [TEST_GALAXY_PATH, TRAINING_GALAXY_PATH, CLASSIFIERS_PATH, RESULT_SOLUTION_DIR]

    for path in changingPaths:
        if not os.path.exists(path):
            os.makedirs(path)

    descriptionFile = open(RUN_DESCRIPTION, 'w')
    descriptionFile.write(description)


