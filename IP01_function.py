import numpy as np

def coCoeffIt(n,it):
    sx1 = 0.0
    sx2 = 0.0
    sx1x2 = 0.0
    sx1sq = 0.0
    sx2sq = 0.0
    #iter = np.nditer([I1[:,:],I2[:,:]],op_flags = ["readonly"])
    for x1,x2 in it:
        sx1 += np.sum(x1)
        sx2 += np.sum(x2)
        sx1x2 += np.sum(x1*x2)
        sx1sq += np.sum(x1*x1)
        sx2sq += np.sum(x2*x2)

    counter = sx1x2 - 1.0/n*sx1*sx2
    denominator = 1.0 /(n*n) * (n*n*sx1sq*sx2sq - n*sx1sq*sx2*sx2
                                - n*sx2sq*sx1*sx1 + sx1*sx1*sx2*sx2)
    return counter / np.sqrt(denominator)

def correlation(img,mask):
    rows,cols = mask.shape
    n = float(rows*cols)
    rI,cI = img.shape
    imgCorr = np.zeros((rI,cI),np.float32)
    startC = int(round( cols/2 ))
    startR = int(round( rows/2 ))

    r = startR
    c = startC
    for corrC in np.nditer(imgCorr[startR:rI - startR, startC:cI - startC], op_flags =["writeonly"]):
        it = np.nditer([img[r-startR : r+startR , c-startC : c+startC], mask[:,:]], flags=["buffered","external_loop"],op_flags =["readonly"], op_dtypes=["float64","float64"])
        corrC[...] = coCoeffIt(n,it)
        it.reset()
        c+=1
        if c >= cI-startC:
            c = startC
            r+=1
    return imgCorr

'''
# slow  version of correlation
def correlation2(img,mask):

    rows,cols = mask.shape

    n = float(rows*cols)

    rI,cI = img.shape

    imgCorr = numpy.zeros((rI,cI),numpy.float32)

    startC = int(round( cols/2 ))
    startR = int(round( rows/2 ))

    r = startR
    c = startC

    for r in range(startR,rows-startR):
        for c in range(startC, cols-startC):
             imgCorr[r-startR:r + startR, c-startC : c+startC] = coCoeff(img[r-startR: r+startR, c-startC: c+startC],mask[:])

    return imgCorr
'''

def getMaximumCorrPoint(corrImg):
    maxIndex = np.argmax(corrImg[:,:])
    rows,cols = corrImg.shape
    c = maxIndex % cols
    r = int(maxIndex / cols)
    return (r,c)

