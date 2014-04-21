cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport sin, cos, round, ceil, pow, tan, abs
from cpython cimport bool

ctypedef np.uint8_t DTYPE_t

cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b
cdef inline double d_max(double a, double b): return a if a >= b else b
cdef inline double d_min(double a, double b): return a if a <= b else b

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline np.uint32_t c_sum2D(DTYPE_t [:,:] arr) nogil:
    cdef np.uint32_t result = 0
    cdef unsigned int i = 0, j = 0
    cdef unsigned int width = arr.shape[0]
    cdef unsigned int height = arr.shape[1]
    
    for i in range(width):
        for j in range(height):
            result = <unsigned int> result + arr[i, j]
            
    return result


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def countInWindowEst(np.ndarray[DTYPE_t, ndim=2] img, int dim):
    cdef DTYPE_t [:,:] img_view = img
    cdef unsigned int width = img.shape[0] - dim
    cdef unsigned int height = img.shape[1] - dim
    cdef np.ndarray[DTYPE_t, ndim=2] result = np.zeros([width, height], dtype=np.uint8)
    cdef DTYPE_t [:,:] result_view = result
    cdef unsigned int i = 0, j = 0
    cdef unsigned int pointSum = 0
    
    for i in range(width):
        for k in range(height):
            result_view[i, k] = <DTYPE_t>(c_sum2D(img_view[i:i+dim,k:k+dim])/pow(dim,2))
    
    return result


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def generateFlattenedDiskImg(np.ndarray[DTYPE_t, ndim=2] img, int centerX, int centerY, float radius):
    cdef DTYPE_t [:,:] img_view = img
    cdef float rotPosX = 0
    cdef float rotPosY = 0
    cdef np.ndarray[np.float_t, ndim=1] thetaRange = np.arange(0, 2*np.pi, 0.01, dtype=np.float)
    cdef np.ndarray[DTYPE_t, ndim=2] result = np.zeros([thetaRange.shape[0], <unsigned int>ceil(radius)], dtype=np.uint8)
    cdef DTYPE_t [:,:] result_view = result
    cdef float slope = 0
    cdef np.ndarray[np.float_t, ndim=1] xStepRange = np.arange(0,1.0, dtype=np.float)
    cdef unsigned int x=0, y=0
    cdef float xPos = 0

    if centerX < radius or centerY < radius or img.shape[0] < centerX + radius or img.shape[1] < centerY + radius or radius < 5:
        return result

    for x, theta in enumerate(thetaRange):
        rotPosX = cos(theta)*radius
        rotPosY = sin(theta)*radius

        if rotPosX == 0:
            continue

        slope = rotPosY/rotPosX

        xPos = 0
        for stepPos in range(<unsigned int>radius):
            xPos = stepPos*rotPosX/radius
            result_view[x, stepPos] = <DTYPE_t>img_view[<unsigned int>round(xPos+centerX), <unsigned int>round(slope*xPos+centerY)]

    return result


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def doLineTest(np.ndarray[DTYPE_t, ndim=2] img):
    cdef np.ndarray[DTYPE_t, ndim=2] scratch_img = img.copy()
    cdef DTYPE_t [:,:] img_view = scratch_img
    cdef unsigned int imgHeight = img.shape[1]
    cdef unsigned int imgWidth = img.shape[0]
    cdef unsigned int i=0, j=0, k=0

    #Skip bulge area variables
    cdef unsigned int startYPos = 0
    cdef bool foundStartPos = False
    cdef unsigned int rowSum = 0
    cdef unsigned int rowAvg = 0

    #Skip the bulge area
    while startYPos < imgHeight:
        rowSum = 0
        for i in xrange(imgWidth):
            rowSum += <unsigned int>img_view[i, startYPos]
        rowAvg = <unsigned int>rowSum / imgWidth

        if rowAvg < 127:
            break

        startYPos += 1

    scratch_img = scratch_img[:,startYPos:]
    img_view = scratch_img
    imgHeight = img_view.shape[1]
    imgWidth = img_view.shape[0]

    #Do the line test
    cdef np.ndarray[np.double_t, ndim=1] thetaRange = np.arange(-np.pi/2 + .1, np.pi/2 - .1, np.pi/150)
    cdef np.ndarray[np.uint32_t, ndim=1] runs = np.zeros_like(thetaRange, dtype=np.uint32)
    cdef np.uint32_t[:] run_view = runs
    cdef double theta = 0
    cdef double maxOffset = 0, minOffset = 0, offset = 0
    cdef double slope = 0
    cdef unsigned int startX = 0, startY = 0, endX = 0, endY = 0
    cdef unsigned int runLength = 0
    cdef unsigned int x = 0, y = 0

    if imgHeight > 0:
        for i, theta in enumerate(thetaRange):
            slope = tan(theta)
            if slope == 0:
                continue

            if slope <= 0:
                maxOffset = imgHeight - slope*imgWidth
                minOffset = 0
            else:
                maxOffset = imgHeight
                minOffset = -1*slope*imgWidth

            for j in range(<unsigned int>((maxOffset - minOffset)/abs(slope*2))):
                offset = minOffset + j*abs(slope*2)

                if slope <= 0:
                    startX = <unsigned int> round(d_max(0, (imgHeight - offset)/slope))
                    endX = <unsigned int> round(d_min(imgWidth, -1*offset/slope))
                    startY = <unsigned int> round(d_min(imgHeight, offset))
                    endY = <unsigned int> round(d_max(0, imgWidth*slope + offset))

                elif slope > 0:
                    startX = <unsigned int> round(d_max(0, -1*offset/slope))
                    endX = <unsigned int> round(d_min(imgWidth, (imgHeight - offset)/slope))
                    startY = <unsigned int> round(d_max(0, offset))
                    endY = <unsigned int> round(d_min(imgHeight, imgWidth*slope + offset))

                runLength = 0
                if theta > -1*np.pi/4 and theta < np.pi/4:
                    for x in range(startX, endX):
                        y = <unsigned int>(slope*x+offset)
                        if y < imgHeight:
                            if img_view[x, y] > 255*0.25:
                                runLength += 1
                                if runLength > runs[i]:
                                    runs[i] = runLength
                            else:
                                runLength = 0
                else:
                    for y in range(int_min(startY, endY), int_max(startY, endY)):
                        x = <unsigned int>((y-offset)/slope)
                        if x < imgWidth:
                            if img_view[x, y] > 255*0.25:
                                runLength += 1
                                if runLength > runs[i]:
                                    runs[i] = runLength
                            else:
                                runLength = 0

    return runs


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def normalizeImageStart(np.ndarray[DTYPE_t, ndim=2] img):
    cdef DTYPE_t [:,:] img_view = img
    cdef unsigned int matchScore = 0
    cdef unsigned int index1 = 0
    cdef unsigned int index2 = 0
    cdef unsigned int maxScore = 0
    cdef unsigned int maxScorePos = 0
    cdef unsigned int i=0
    cdef unsigned int pix
    cdef unsigned int score

    for i in range(img_view.shape[0]):
        index1 = i
        index2 = <unsigned int>(index1 + img_view.shape[0]/2)
        if index2 >= img_view.shape[0]:
            index2 = <unsigned int>(index2 - img_view.shape[0])

        score = 0
        for pix in img_view[index1]:
            if pix < 127:
                break

            score += <unsigned int>pix

        for pix in img_view[index2]:
            if pix < 127:
                break

            score += <unsigned int>pix

        if score > maxScore:
            maxScore = score
            maxScorePos = i

    cdef np.ndarray[DTYPE_t, ndim=2] result_img = np.zeros_like(img)
    cdef DTYPE_t [:,:] result_img_view = result_img

    result_img_view[0:img_view.shape[0] - maxScorePos] = img_view[maxScorePos:img_view.shape[0]]
    result_img_view[img_view.shape[0] - maxScorePos:img_view.shape[0]] = img_view[0:maxScorePos]

    return result_img

