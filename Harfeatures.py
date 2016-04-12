#HRV features

import numpy as np
from math import log

def DFALeastSquare(y, x=None):    
    if x==None:
        x = np.arange(len(y))
    x = np.array(x)
    y = np.array(y)
    Sxy = np.sum(x*y) - np.sum(x)*np.sum(y)/len(y)
    Sxx = np.sum(x*x) - np.sum(x)*np.sum(x)/len(x)
    x_bar = np.sum(x)/len(x)
    y_bar = np.sum(y)/len(y)
    b = Sxy/Sxx
    a = y_bar - b*x_bar
    return a+b*x

def PNNx(rrIntervalSeries, x = 20):
    rr1 = np.array(rrIntervalSeries[1:])
    rr2 = np.array(rrIntervalSeries[:-1])
    factor = x/1000
    count = len([x for x in (rr1 - rr2) if x>=factor])
    return count/len(rrIntervalSeries)

def SDNN(rrIntervalSeries):
    return np.std(rrIntervalSeries, ddof=1)

def RMSSD(rrIntervalSeries):
    rr1 = np.array(rrIntervalSeries[1:])
    rr2 = np.array(rrIntervalSeries[:-1])
    diff = np.array([x for x in (rr1 - rr2)])
    return np.sqrt(np.sum(diff**2)/len(diff))
    
def SDSD(rrIntervalSeries):
    rr1 = np.array(rrIntervalSeries[1:])
    rr2 = np.array(rrIntervalSeries[:-1])
    diff = np.array([x for x in (rr1 - rr2)])
    return np.std(diff, ddof=1)

def SD1SD2(rrIntervalSeries):
    rr1 = np.array(rrIntervalSeries[:-1])
    rr2 = np.array(rrIntervalSeries[1:])
    d1 = np.abs(rr1 - rr2)/np.sqrt(2)
    d2 = np.abs(rr1 + rr2 - np.mean(rrIntervalSeries))/np.sqrt(2)
    sd1 = np.std(d1, ddof=1)
    sd2 = np.std(d2, ddof=1)
    return sd1/sd2

def DFA(rrIntervalSeries, n = 3):        # detrended fluctuation analysis
    rLen = len(rrIntervalSeries)
    windowSize = rLen/n
    rr_bar = np.mean(rrIntervalSeries)
    yk = np.array([np.sum(rrIntervalSeries[:k+1])-(k+1)*rr_bar for k in range(len(rrIntervalSeries))])
    ynk = [DFALeastSquare(yk[windowSize*i:windowSize*(i+1)]) for i in range(n)]
    ynkarray = np.array(ynk).reshape((1, -1))
    fn = np.sqrt(np.sum((yk-ynkarray)**2)/rLen)
    return fn
    
def HRVTriangularIndex(rrIntervalSeries):
    rrmin = np.min(rrIntervalSeries)
    rrmax = np.max(rrIntervalSeries)
    binSize = rrmax - rrmin

    binSizelist = [list(np.abs(np.array(rrIntervalSeries[i+1:])-rrIntervalSeries[i])) for i in range(len(rrIntervalSeries)-1)]
    while 1:
        if 10e-7<binSizelist[-1][-1]<binSize:
            binSize = binSizelist[-1][-1]
            break
        else:
            if len(binSizelist[-1])==1:
                binSizelist.pop()
            else:
                binSizelist[-1].pop()
    noBins = (rrmax-rrmin)/binSize
    bins = [0]*(int(noBins)+1)
    for i in range(len(rrIntervalSeries)):
        distance = rrIntervalSeries[i] - rrmin
        lowerLim = distance/binSize
        bins[int(lowerLim)] += 1

    hti = len(rrIntervalSeries)/max(bins)
    return hti

def CTM(rrIntervalSeries):
    rr = rrIntervalSeries
    rlist = [np.sqrt((rr[x+2]-rr[x+1])**2+(rr[x+1]-rr[x])**2) for x in range(len(rr)-2)]
    r = (max(rlist) - min(rlist)) / 8
    ctm = np.sum(np.array(rlist)<r)
    ctm = ctm/(len(rr)-2)
    return ctm

def SpatialFillingIndex(rrIntervalSeries, n=10):
    aMatrix = np.array([rrIntervalSeries[:-1],rrIntervalSeries[1:]]).T
    bMatrix = aMatrix / np.max(aMatrix)
    cMatrix = np.zeros((n, n))
    cAxis = [[-1+i*(2/n),-1+(i+1)*(2/n)] for i in range(n)]
    cAxis[-1][1] += 10e-7
    for i in range(len(bMatrix)):
        cAxisx1 = cAxisy1 = np.array([x[0] for x in cAxis])
        cAxisx2 = cAxisy2 = np.array([x[1] for x in cAxis])
        cAxisx1 = np.where(cAxisx1 <= bMatrix[i][0], True, False)
        cAxisx2 = np.where(cAxisx2 >  bMatrix[i][0], True, False)
        cAxisy1 = np.where(cAxisy1 <= bMatrix[i][1], True, False)
        cAxisy2 = np.where(cAxisy2 >  bMatrix[i][1], True, False)
        cMatrix[cAxisy1 & cAxisy2, cAxisx1 & cAxisx2] += 1
    s = np.sum((cMatrix/np.sum(cMatrix))**2)
    yita = s / (n**2)
    return yita

def ApEnm(rrIntervalSeries, m, r):
    size = len(rrIntervalSeries)
    rr = rrIntervalSeries

    # 利用m个连续的rr值组成(size-m)*m的数组
    tempRR = [rr[i:i+m] for i in range(size-m+1)]
    tempRR = np.array(tempRR)

    # 按照算法计算CMI
    CMI = [sum(np.max(abs(tempRR - i),axis=1)<=r) for i in tempRR]
    CMI = np.array(CMI) / len(tempRR)

    # 计算CMI的平均值
    ApEnm = sum(CMI) / len(CMI)

    return ApEnm

def ApEn(rrIntervalSeries, m, r):
    apen1 = ApEnm(rrIntervalSeries, m, r)
    apen2 = ApEnm(rrIntervalSeries, m+1, r)
    apen = np.log(apen1 / apen2)
    return apen

def Hrvfeatures(rrIntervalSeries):
    r = np.std(rrIntervalSeries) # 用于计算ApEn
    average = np.mean(rrIntervalSeries)
    pnn20 = PNNx(rrIntervalSeries, x=20)
    pnn50 = PNNx(rrIntervalSeries, x=50)
    sdnn = SDNN(rrIntervalSeries)
    rmssd = RMSSD(rrIntervalSeries)
    sdsd = SDSD(rrIntervalSeries)
    sd1sd2 = SD1SD2(rrIntervalSeries)
    hti = HRVTriangularIndex(rrIntervalSeries)
    ctm = CTM(rrIntervalSeries)
    sfi = SpatialFillingIndex(rrIntervalSeries)
    apen1 = ApEn(rrIntervalSeries, 1, 0.1*r)
    apen2 = ApEn(rrIntervalSeries, 1, 0.15*r)
    apen3 = ApEn(rrIntervalSeries, 1, 0.2*r)
    apen4 = ApEn(rrIntervalSeries, 1, 0.25*r)
    # DFA
    #fn15 = DFA(rrIntervalSeries, 15)   value is too small
    fn10 = DFA(rrIntervalSeries, 10)
    fn6 = DFA(rrIntervalSeries, 6)
    fn5 = DFA(rrIntervalSeries, 5)
    fn3 = DFA(rrIntervalSeries, 3)
    fn2 = DFA(rrIntervalSeries, 2)
    x = [log(3), log(5), log(6), log(10), log(15)]
    y = [log(fn10), log(fn6), log(fn5), log(fn3), log(fn2)]
    dfay = DFALeastSquare(y, x)
    dfa = (dfay[1]-dfay[0])/(log(5)-log(3))
    
    # return [pnn20, pnn50, sdnn, rmssd, hti, ctm, sfi, apen1, apen2, apen3, apen4]
    return [average, pnn20, pnn50, sdnn, rmssd, sdsd, sd1sd2, dfa, hti, ctm, sfi, apen1]
