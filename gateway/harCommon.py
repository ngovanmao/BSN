#!/usr/bin/python
import numpy as np
import math
import scipy.stats
import multiprocessing
import os
import time

noActivities = 5
noFullSensor = 9
noSensor = noFullSensor 
dRate = 8
samplingRate = 32 # 32 or 64Hz
startIndex = 1
endIndex = (samplingRate * noSensor) / dRate + 1 #96 + 1 # index of sensor timestamp. last number is 48 + 1.
lastIndex = endIndex + 2 # 27
ZIGBEE_OFFSET = 8
SENSORLEN = endIndex + ZIGBEE_OFFSET 
DB_DIR = "dataBase/"
# plus 8 for header of Zigbee stuff, 1 for sensortime 
						  # 33 if  32Hz /8; 57 if 32Hz/8
windowSize = 2 # 2s is a window size
packetWindow = dRate * windowSize
# Obtain data from database for training purpose
# For BLE-mote
#LHID = 9430 ; #RHID = 13739; #LFID = 63162; #RFID = 64591
# For Sensor tag
CID = 42337 #dummy value now
LHID = 15622; RHID = 50569; LFID = 53764; RFID = 9346
ID = np.array([CID, LHID, RHID, LFID, RFID])

FULL_GYRO = 500.0 # deg/s
FULL_ACC = 16.0 
FULL_MAG = 4912

SAMPLING_RATE_ACC = 30 # Hz
SAMPLING_RATE_TEM = 2  # Hz
SAMPLING_RATE_ECG = 32 # Hz
PACKET_RATE_ACC = 3
PACKET_RATE_TEM = 1
PACKET_RATE_ECG = 2

#MAC_TEM = 40705
MAC_TEM = 40695
MAC_ECG = 42398
MAC_ACC = 40682

STATE_NORMAL = 0
STATE_URGENT = 1
STATE_EXTENDED = 2

img = ['../picture/sitting.png', '../picture/running.png',\
        '../picture/walking.png', '../picture/standing.png'] 

#img = ['../Picture/running.png', '../Picture/walking.png', \
#        '../Picture/sitting.png', '../Picture/standing.png', '../Picture/lying.png','../Picture/idle.png']
convertMatrixAcc = np.array([32768.0/FULL_ACC, 32768.0/FULL_ACC, 32768.0/FULL_ACC])

convertMatrix = np.array([32768.0/FULL_ACC, 32768.0/FULL_ACC, 32768.0/FULL_ACC,\
                          65536.0/FULL_GYRO,65536.0/FULL_GYRO,65536.0/FULL_GYRO])

convertMatrixFull = np.array([32768.0/FULL_ACC, 32768.0/FULL_ACC, 32768.0/FULL_ACC,\
                          65536.0/FULL_GYRO,65536.0/FULL_GYRO,65536.0/FULL_GYRO,\
                          1,1,1])

def hStackArray(data, add):
    """Function: Stack matrix add to data with horizontal direction
    Output: data with added matrix
    """
    if len(data) > 0:
        data = np.hstack((data, add))
    else:
        data = add
    return data

def vStackArray(data, add):
    """Function: Stack matrix add to data with vertical direction"
    Output: data with added matrix
    """
    if len(data) > 0:
        data = np.vstack((data, add))
    else:
        data = add
    return data

def vStackArrays(data, add, maxSize):
    """Function: Stack matrix add-matrix to data with vertical direction.      
              Append add-matrix to data. if matrix after added has bigger size
              than maxSize, the oldest part will be dropped                 
    Input   : data, adding matrix, maxSize of data can be stored
    Output: data with added matrix
    """
    if len(data) == 0:
        data = add
    elif (len(data) + len(add)) < maxSize:
        data = np.vstack((data, add))
    else:
        data = np.vstack((data, add))
        data = data[(len(data) - maxSize):] 
    return data

def span(func, ls):
    """returns a tuple
    first element is longest array of elements that satisfy the predicate func
    second element is everything after that
    """
    if len(ls)==0:
        return (np.array([]),np.array([]))
    else:
        head = ls[0]
        tail = ls[1:]
        if(func(head)):
            ys, zs = span(func, tail)
            return (np.insert(ys, 0, head), zs)
        else:
            return (np.array([]),ls)

def groupBy(func, array):
    """Groups the elements, using the given func as a test for equality
    """
    if(len(array)==0):
        return []
    head = array[0]
    tail = array[1:]
    ys, zs = span((lambda b:func(head,b)),tail)
    ls = np.insert(ys,0,head)
    res = groupBy(func, zs)
    res.insert(0,ls.tolist())
    return res


def splitWhen(func, ls):
    """Function: Seperates a list into a tuple of two lists - split at the first
              element where the predicate did not hold between it and the previous
              element
    Input:    A function that takes two elements and returns a bool.
              List to be split
    Output:   ([a],[a]) for some type a
              (ls,[]) If predicate held for all consecutive elements.
    """
    prevElem = ls[0]
    idx = 0
    for elem in ls[1:]:
        idx+=1
        if (func(prevElem,elem)):
            return (ls[:idx],ls[idx:])
        prevElem = elem
    return (ls,[])


def clusterWhen(func, ls):
    """Function: Separates a list into a lists of lists (clusters) such that a 
              predicate is satisfied for any two consecutive elements in a
              cluster
    Input:    Function that takes two elements and returns a bool.
              List to be clustered.
    Output:   A cluster of lists, with the original list split at the points
              where the predicate did not hold between consecutive elements.
    """
    clusters = []
    curList = ls
    while True:
        fst, snd = splitWhen(lambda a,b: not func(a,b),curList)
        clusters.append(fst)
        if snd == []:
            break
        else:
            curList = snd 
    return clusters


def roundUpRawDataBySysTime(X):
    """Function: Rounds up raw data to have 8 packets every second - checks for
              multiple recording sessions and adjusts accordingly 
    Input   : data, adding matrix, maxSize of data can be stored      
    Output: data with added matrix                                          
    """
    # round down to 64 times
    # Sort based on systemtime
    X = X[X[:, 0].argsort()]
    
    Xgroups = clusterWhen((lambda packet1,\
              packet2:packet2[endIndex]>=packet1[endIndex]),X.tolist())
    print 'Number of collection clusters identified', len(Xgroups)
    XXgroups = map(lambda X:roundUpRawDataTrain(np.array(X)), Xgroups)
    XX = [inner 
            for outer in XXgroups
                for inner in outer]
    XX = np.array(XX)
    #MUST sort according to sensor time again, as later function relies on it
    #to extract feature vector
    XX = XX[XX[:,endIndex].argsort()]
    return XX


def roundUpRawDataTrain(X):
    # round down to 64 times
    # Sort based on sensortime 
    X = X[X[:, endIndex].argsort()]
    # Get the first sensortime
    t = X[0, endIndex]
    # Get the last sensortime
    tmax = X[len(X) - 1, endIndex]
    XX = np.array([])
    # Check every second (sensortime) and fulfil the missed packets 
    # with mean the received data
    while (t <= tmax):
        idx = np.where(X[:, endIndex] == t)
        temp = X[idx]
        meanTemp = np.mean(temp, axis = 0)
        lenT = len(temp)
        if lenT >= 1 and lenT < dRate:
            print 'missing some data, filling up'
            print t, tmax
            for i in range(dRate - lenT):
                temp = vStackArray(temp, meanTemp)
            XX = vStackArray(XX, temp)
        elif lenT == dRate:
            XX =vStackArray(XX, temp)
        else:
            pass
            # There are lots of missing data?
            # print 't = ', t, 'lenT = ', lenT
        t += 1
    return XX


def roundUpRawData(X):
    """Function: Each window = 2second, that means we have two timestamps in data
              This function is to round up the missing samples within window
    Input: Data in window
    Output: Full data for that window
    """
    # round down to 64 times
    X = X[X[:, endIndex].argsort()]
    t = X[0, endIndex]
    tmax = X[len(X) - 1, endIndex]
    XX = np.array([])
    if t == tmax:
        print '*************Missing too much************'
        X = vStackArray(X,X)
        #return XX
    while (t <= tmax):
        idx = np.where(X[:, endIndex] == t)
        temp = X[idx]
        meanTemp = np.mean(temp, axis = 0)
        if len(temp) < dRate and len(temp) > 1:
            for i in range(dRate - len(temp)):
                temp = vStackArray(temp, meanTemp)
            XX = vStackArray(XX, temp)
        elif len(temp) == dRate:
            XX =vStackArray(XX, temp)
        else:
            print 'Missing too much. Len = ', len(temp)
        t += 1
    return XX


def distanceR(temp):
    """Function: calculate the amplitude of X, Y, Z accelerometer sensors
    Output: R =sqrt(X^2 + Y^2 + Z^2)                                  
    """
    return math.sqrt(temp[0]*temp[0]+temp[1]*temp[1]+temp[2]*temp[2])

def convertDataEcg(X, numSen):
    """Function: convert raw sensor data to meaningful sensor data             
    For example: acc_x = 63896; then convert to acc_x = -1.64               
    Input: full raw all sensors data accelerometer
    Output:                                                                 
            matrix with the same size as input                              
            [ax, ay, az, sensortime]            
    """
    t = np.int16(X[:,:-1])
    d = np.array([], dtype=np.int16)
    for i in range(len(t)):
        # sampling rate 32Hz, 1 packet/s
        for j in range(SAMPLING_RATE_ECG/ PACKET_RATE_ECG): 
            temp = t[i,numSen * j : numSen * (j+1)]
            temp = hStackArray(temp, X[i,-1])
            d = vStackArray(d, temp)
    return d

def convertDataTem(X, numSen):
    """Function: convert raw sensor data to meaningful sensor data             
    For example: acc_x = 63896; then convert to acc_x = -1.64               
    Input: full raw all sensors data accelerometer
    Output:                                                                 
            matrix with the same size as input                              
            [ax, ay, az, sensortime]            
    """
    # Get only sensor raw values, remove systemtime
    t = np.int16(X[:,:-1])
    d = np.array([], dtype=np.int16)
    for i in range(len(t)):
        for j in range(SAMPLING_RATE_TEM / PACKET_RATE_TEM): # sampling rate 2Hz, 1 packet/s
            temp = t[i,numSen * j : numSen * (j+1)] / 100.00
            temp = hStackArray(temp, X[i,-1])
            d = vStackArray(d, temp)
    return d

def convertDataAcc(X, numSen):
    """Function: convert raw sensor data to meaningful sensor data             
    For example: acc_x = 63896; then convert to acc_x = -1.64               
    Input: full raw all sensors data accelerometer
    Output:                                                                 
            matrix with the same size as input                              
            [ax, ay, az, sensortime]            
    """
    # Get only sensor raw values, remove sensortime/label
    t = np.int16(X[:,:-1])
    d = np.array([], dtype=np.int16)
    for i in range(len(t)):
        for j in range(SAMPLING_RATE_ACC / PACKET_RATE_ACC):
            temp = t[i,numSen * j : numSen * (j+1)] / convertMatrixAcc
            #Add back sensor time
            temp = hStackArray(temp, X[i,-1])
            d = vStackArray(d, temp)
    return d

"========================================================================="
" Function: convert raw sensor data to meaningful sensor data             "
" For example: acc_x = 63896; then convert to acc_x = -1.64               "
" Input: full raw all sensors data (accelerometer, gyroscope, magnetometer)"
" Output:                                                                 "
"         matrix with the same size as input                              "
"             [ax, ay, az, gx, gy, gz, mx, my, mz, sensortime]            "
"========================================================================="
def convertData328(X, numSen):
    # Get only sensor raw values, remove systemtime
    t = np.int16(X[:,startIndex:-1])
    d = np.array([], dtype=np.int16)
    global convertMatrix 
    if numSen == 9:
        convertMatrixTemp = convertMatrixFull
    else: 
        convertMatrixTemp = convertMatrix
    for i in range(len(t)):
        for j in range(samplingRate / dRate):
            temp = t[i,numSen * j : numSen * (j+1)] / convertMatrixTemp
            temp = hStackArray(temp, X[i,-1])
            d = vStackArray(d, temp)
    return d

"========================================================================="
" Function: convert raw sensor data to meaningful sensor data             "
" For example: acc_x = 63896; then convert to acc_x = -1.64               "
" Input: full raw all sensors data (accelerometer, gyroscope, magnetometer)"
" Output:                                                                 "
"         matrix with the same size as input                              "
"             [ax, ay, az, gx, gy, gz, mx, my, mz, sensortime]            "
"========================================================================="
def convertData(X):
    #X = np.asarray(X)
    #X = np.int_(X)
    # Get only sensor raw values, remove systemtime
    print 'This is depricated API, please use convertData328(X, numSens)'
    t = np.int16(X[:,startIndex:endIndex+1])
    d = np.array([], dtype=np.int16)
    global convertMatrix 
    numSen = 6
    if numSen == 9:
        convertMatrixTemp = convertMatrixFull
    else: 
        convertMatrixTemp = convertMatrix
    for i in range(len(t)):
        for j in range(samplingRate / dRate):
            #tempIndex = range(startIndex+3*j, startIndex+3*(j+1)) + range(endIndex, 26)
            #temp = t[i,startIndex + noSensor * j : startIndex + noSensor * (j+1)] / convertMatrix
            # fix misorder with Duong.
            temp = t[i,noSensor * j : noSensor * (j+1)] / convertMatrixTemp
            #temp = hStackArray(temp, distanceR(temp))
            temp = hStackArray(temp, X[i,endIndex])
            d = vStackArray(d, temp)
    return d

def chopData(T, numSen):
    chopTemp = np.array([])
    if numSen == 9:
        convertMatrixTemp = convertMatrixFull
    else: 
        convertMatrixTemp = convertMatrix
    # Get only sensor raw values
    t = np.int16(T[startIndex:endIndex+1])
    for j in range(samplingRate / dRate):
        temp = t[noSensor * j : noSensor * (j+1)] / convertMatrixTemp
        # Add sensortime and label data
        temp = hStackArray(temp, T[endIndex:lastIndex])
        chopTemp = vStackArray(chopTemp, temp)
    return chopTemp


"========================================================================="
" Function: Parallel  convert raw sensor data to meaningful sensor data   "
" For example: acc_x = 63896; then convert to acc_x = -1.64               "
" Input : raw sensor data with multi-dimensional sensor data              "
" Output:                                                                 "
"         matrix with the same size as input [ax, ay, az, gx, gy, gz, sensortime, label]    "
"========================================================================="
def convertDataParallel(X, numSen):
    numCPU = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(numCPU)
    X = np.int_(X)
    d = np.array([])
    tasks = []
    i=0
    # Build a list of task
    while i < len(X):
        tasks.append(X[i])
        i += 1
    # Run task in parallel
    tstart = time.time()*1000.0
    results = [pool.apply_async( chopData, (t,numSen)) for t in tasks]
    d = recursiveVStack(results, lambda x: x.get())
    print 'Conversion time: ', time.time()*1000.0 - tstart
    pool.close()
    pool.join()
    return d
#Takes array of arrays, stacks arrays on top of each other
"""
X1=np.array([1,2,3])
X2=np.array([1,2,4])
X3=np.array([1,2,5])
X4=np.array([1,2,6])
X5=np.array([1,2,7])
X=[X1,X2,X3,X4]
Y=[X1,X2,X3,X4,X5]
recursiveVStack(X)
results in: array([[1, 2, 3],
       [1, 2, 4],
       [1, 2, 5],
       [1, 2, 6]])
recursiveVStack(Y)
results in: array([[1, 2, 3],
       [1, 2, 4],
       [1, 2, 5],
       [1, 2, 6],
       [1, 2, 7]])

Second argument is a function that will be applied to each individual array before stacking.
If not supplied, no function will be applied
"""
def recursiveVStack(X, f = lambda x:x):
    if len(X)==1:
        return f(X[0])
    first = X[:len(X)/2]
    second = X[len(X)/2:]
    firstStacked = recursiveVStack(first, f)
    secondStacked = recursiveVStack(second, f)
    return vStackArray(firstStacked, secondStacked)

"========================================================================="
" This function define the extracted features                             "
" Input: X is converted sensor data                                       "
" Output: [Min, Max, Mean, Variance, Skew, Kurtosis, 5 peaks of FFT]      "
"          within window                                                  "
"         If we don't use FFT, each dimension sensor extracts to 6 features"
"========================================================================="
def feature_unit(X):
    #weird issue: some packets are only 32 units long! means autocorrelation,
    #which varies in length based on the size of the packet, will not be a good
    #feature
    try:
       #return np.array([np.min(X), np.max(X), np.mean(X),
       #         np.var(X), scipy.stats.skew(X), scipy.stats.kurtosis(X)])
        #autocor = autocorr(X)
        #if len(autocor)==32:
        #    autocor = np.concatenate((autocor,autocor))
        #if len(autocor)>64:
         #   autocor = autocor[:64]
        #if len(autocor)!=64:
        #    print autocor, len(autocor)
        #    while len(autocor)<64:
         #       np.append(autocor, [0]) 
        return np.array([np.min(X), np.max(X), np.mean(X),
                 np.var(X), scipy.stats.skew(X), scipy.stats.kurtosis(X)])
                    #firstFivePeaksOfFFT(X)))
    except ValueError:
        print 'ValueError in feature_unit ', X

num = 0
def firstFivePeaksOfFFT(X):
    global num
    w = abs(np.fft.rfft(X))
    fft_freq = np.fft.fftfreq(len(X))
    freq = fft_freq #* 1.0/windowSize
    peaks,throughs = peakDetect(w,1)
    if(len(peaks)!=0):
        peaks = peaks[np.argsort(peaks[:,1])]
    peaks = peaks[::-1]
    first_five = []
    count = 0
    for peak in peaks[1:]:
        peak_index = peak[0]
        first_five.append(freq[peak_index])
        count+=1
        if(count == 5):
            break
    while(len(first_five)<5):
        first_five.append(0)
    #DEBUG
    #if num <=10 or len(first_five)<5:
    #    print first_five    
    #num +=1
    #DEBUG
    first_five.sort()
    return np.array(first_five)
def magnitude(x, y, z):
    return (x**2+y**2+z**2)**(0.5)

def extract_feature_acc(sensor, numSen):
    AX = feature_unit(sensor[:,0])
    AY = feature_unit(sensor[:,1])
    AZ = feature_unit(sensor[:,2])
    return np.hstack((AX, AY, AZ))

def extract_feature(sensor, numSen):
    AX = feature_unit(sensor[:,0])
    AY = feature_unit(sensor[:,1])
    AZ = feature_unit(sensor[:,2])
    GX = feature_unit(sensor[:,3])
    GY = feature_unit(sensor[:,4])
    GZ = feature_unit(sensor[:,5])
    if numSen == 6:
        return np.hstack((AX, AY, AZ, GX, GY, GZ))
    else:
        MX = feature_unit(sensor[:,6])
        MY = feature_unit(sensor[:,7])
        MZ = feature_unit(sensor[:,8])
        return np.hstack((AX, AY, AZ, GX, GY, GZ, MX, MY, MZ))

from numpy import NaN, Inf, arange, isscalar, asarray, array

def peakDetect(ls, delta, x = None):
    maxtab = []
    mintab = []
       
    if x is None:
        x = arange(len(ls))
    
    ls = asarray(ls)
    
    if len(ls) != len(x):
        sys.exit('Input vectors ls and x must have same length')
    
    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    
    if delta <= 0:
        sys.exit('Input argument delta must be positilse')
    
    mn, mx = Inf, -Inf #mn is min, mx is max
    mnpos, mxpos = NaN, NaN
    
    lookformax = True
    
    for i in arange(len(ls)):
        this = ls[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        
        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return array(maxtab), array(mintab)


"========================================================================="
''' Input: S = [AX, AY, AZ, label] '''
"========================================================================="
def get_feature_acc(S, numSen, numLabel):
    S_fe = np.array([])
    lS_fe =np.array([])
    for i in range(numLabel):
        print 'Obtaining features for activity ', i 
        # get all data having the same label
        ic = np.where(S[:, -1] == i)
        Si = S[ic]
        lenData = len(Si)
        moduloLenData = lenData % 30
        print 'old len data = ', len(Si)
        if moduloLenData > 0:
            Si = Si[:lenData - moduloLenData, :]
            print 'new len data = ', len(Si)
        # remove the label data
        Si = Si[:,:-1]
        # array contain extracted feature data of each label data
        S_fe_unit = np.array([])
        j = 0
        jct = np.asarray(range(60))
        while j < len(Si) - 30:
            j += 30
            # indecies of data within 2 seconds (window) and having the same label
            Si_temp = Si[jct,:]
            S_fe_temp = extract_feature_acc(Si_temp, numSen)
            try:
                S_fe_unit = vStackArray(S_fe_unit, S_fe_temp)
            except ValueError:
                print S_fe_temp
                raise ValueError
            jct += 30
        #create label according the extracted feature
        lS_fe_temp = np.ones(len(S_fe_unit), dtype=int) * i
        S_fe = vStackArray(S_fe, S_fe_unit)
        lS_fe = np.append(lS_fe, lS_fe_temp)
    return S_fe, lS_fe

"========================================================================="
''' Input: S = [AX, AY, AZ, GX, GY, GZ, sensortime, label] '''
"========================================================================="
def get_feature(S, numSen):
    S_fe = np.array([])
    lS_fe =np.array([])
    for i in range(1, noActivities + 1):
        print 'Obtaining features for activity ', i 
        # get all data having the same label
        ic = np.where(S[:, -1] == i)
        Si = S[ic]
        # remove the label data
        Si = Si[:,:-1]
        # Get the first time stamp
        sTime = Si[0, -1]
        # Get the last time stamp
        sTimeMax = Si[-1, -1]
        # array contain extracted feature data of each label data
        S_fe_unit = np.array([])
        while sTime < sTimeMax:
            # indecies of data within 2 seconds (window) and having the same label
            ict = np.append(np.where(Si[:, -1] == sTime), np.where(Si[:, -1] == sTime+1))
            Si_temp = Si[ict,:]
            if (len(Si_temp) == 0):
                #print sTime
                pass
            else:
                S_fe_temp = extract_feature(Si_temp, numSen)
                try:
                    S_fe_unit = vStackArray(S_fe_unit, S_fe_temp)
                except ValueError:
                    print S_fe_temp
                    raise ValueError
            sTime += 1
        #create label according the extracted feature
        lS_fe_temp = np.ones(len(S_fe_unit), dtype=int) * i
        S_fe = vStackArray(S_fe, S_fe_unit)
        lS_fe = np.append(lS_fe, lS_fe_temp)
    return S_fe, lS_fe


"========================================================================="
" Autocorrelation function                                                "
"========================================================================="
def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size/2:]


"========================================================================="
' Feature extraction'
' For example: size = 5s, data '
" Return indecies of sequence                                             "
"========================================================================="
def slidingWindow(sequence, winSize, step):
    try: it = iter(sequence)
    except TypeError:
        raise Exception("**ERROR** sequence must be iterable.")
    if step > winSize:
        raise Exception("**ERROR** type(winSize) and type(step) must be in int")
    if winSize > len(sequence):
        raise Exception("**ERROR** winSize must not be larger than sequence length")
    # Pre-compute number of chunks to emit
    numOfChunks = ((len(sequence) - winSize)/step) + 1
    # Do the work
    for i in range(0, numOfChunks * step, step):
        yield sequence[i:i + winSize]

def createDatabase328(conn, table):
    """Function: Read sensor from position i with the contains               
               (systemtime, acc_x, acc_y, acc_z, ..., sensortime, activity) 
     Input: i is index position of sensor                                  
     Output: sensor data                                                     
    """
    cursor = conn.cursor()
    sql = "CREATE TABLE %s \
          (systemtime BIGINT NOT NULL, node_id INT,  \
          ax1 INT, ay1 INT, az1 INT, gx1 INT, gy1 INT, gz1 INT, \
          ax2 INT, ay2 INT, az2 INT, gx2 INT, gy2 INT, gz2 INT, \
          ax3 INT, ay3 INT, az3 INT, gx3 INT, gy3 INT, gz3 INT, \
          ax4 INT, ay4 INT, az4 INT, gx4 INT, gy4 INT, gz4 INT, \
          sensortime INT, activity INT DEFAULT NULL)"
    sql2 = "PRAGMA journal_mode=WAL"
    try:
        cursor.execute(sql %(table))
        cursor.execute(sql2)
    except:
        print "Database was already created"
    conn.commit()

def createDatabaseEcg16(conn, table):
    """Function: Create a database for ECG sensor data:
             (x1, x2, x3, ..., x16) 
             each packet contains 16 samples. 
             Sampling rate is initiated as 32Hz.
    """
    cursor = conn.cursor()
    sql = "CREATE TABLE %s \
          (systemtime BIGINT NOT NULL, len INT, time1 INT, time2 INT, state INT, node_id INT, \
          seqno INT, hop INT, rssi INT, \
          x1 INT, x2 INT, x3 INT, x4 INT, x5 INT, x6 INT, \
          x7 INT, x8 INT, x9 INT, x10 INT, x11 INT, x12 INT, \
          x13 INT, x14 INT, x15 INT, x16 INT, \
          mac_stat INT)"
    sql2 = "PRAGMA journal_mode=WAL"
    try:
        cursor.execute(sql %(table))
        cursor.execute(sql2)
    except:
        print "Database ECG sensors was already created"
    conn.commit()

def createDatabaseEcg(conn, table):
    """Function: Create a database for ECG sensor data:
             (x1, x2, x3, ..., x32) 
             each packet contains 32 samples. 
             Sampling rate is initiated as 32Hz.
             state = (scale_sending_rate << 8) + node_schedule_state
             mac_stat = (mac_sent << 8) + mac_lost
    """
    cursor = conn.cursor()
    sql = "CREATE TABLE %s \
          (systemtime BIGINT NOT NULL, len INT, time1 INT, time2 INT, state INT, node_id INT, \
          seqno INT, hop INT, rssi INT, \
          x1 INT, x2 INT, x3 INT, x4 INT, x5 INT, x6 INT, \
          x7 INT, x8 INT, x9 INT, x10 INT, x11 INT, x12 INT, \
          x13 INT, x14 INT, x15 INT, x16 INT, x17 INT, x18 INT, \
          x19 INT, x20 INT, x21 INT, x22 INT, x23 INT, x24 INT, \
          x25 INT, x26 INT, x27 INT, x28 INT, x29 INT, x30 INT, \
          x31 INT, x32 INT, \
          mac_stat INT)"
    sql2 = "PRAGMA journal_mode=WAL"
    try:
        cursor.execute(sql %(table))
        cursor.execute(sql2)
    except:
        print "Database ECG sensors was already created"
    conn.commit()

def createDatabaseAcc(conn, table):
    """Function: Create a database for Accelerometer sensor data:
             (ax1, ay1, az1, ..., ax10, ay10, az10) 
             8 packets/s, each packet contains 8 samples. 
             In total, sampling rate is 30Hz.
    """
    cursor = conn.cursor()
    sql = "CREATE TABLE %s \
          (systemtime BIGINT NOT NULL, len INT, time1 INT, time2 INT, dummy1 INT, node_id INT, \
          seqno INT, hop INT, dummy2 INT, \
          ax1 INT, ay1 INT, az1 INT, \
          ax2 INT, ay2 INT, az2 INT, \
          ax3 INT, ay3 INT, az3 INT, \
          ax4 INT, ay4 INT, az4 INT, \
          ax5 INT, ay5 INT, az5 INT, \
          ax6 INT, ay6 INT, az6 INT, \
          ax7 INT, ay7 INT, az7 INT, \
          ax8 INT, ay8 INT, az8 INT, \
          ax9 INT, ay9 INT, az9 INT, \
          ax10 INT, ay10 INT, az10 INT, \
          sensortime INT)"
    sql2 = "PRAGMA journal_mode=WAL"
    try:
        cursor.execute(sql %(table))
        cursor.execute(sql2)
    except:
        print "Database ADXL346 sensors was already created"
    conn.commit()

def createDatabaseAcc8(conn, table):
    """Function: Create a database for Accelerometer sensor data:
             (ax1, ay1, az1, ..., ax8, ay8, az8) 
             8 packets/s, each packet contains 8 samples. 
             In total, sampling rate is 64Hz.
    """
    cursor = conn.cursor()
    sql = "CREATE TABLE %s \
          (systemtime BIGINT NOT NULL, len INT, time1 INT, time2 INT, dummy1 INT, node_id INT, \
          seqno INT, hop INT, dummy2 INT, \
          ax1 INT, ay1 INT, az1 INT, \
          ax2 INT, ay2 INT, az2 INT, \
          ax3 INT, ay3 INT, az3 INT, \
          ax4 INT, ay4 INT, az4 INT, \
          ax5 INT, ay5 INT, az5 INT, \
          ax6 INT, ay6 INT, az6 INT, \
          ax7 INT, ay7 INT, az7 INT, \
          ax8 INT, ay8 INT, az8 INT, \
          sensortime INT)"
    sql2 = "PRAGMA journal_mode=WAL"
    try:
        cursor.execute(sql %(table))
        cursor.execute(sql2)
    except:
        print "Database ADXL346 sensors was already created"
    conn.commit()

def createDatabaseTem(conn, table):
    """Function: Create a database for temperature and humidity sensor data:
             (tem1 x 100, hum1 x 100, tem2 x 100, hum2 x 100)
    """
    cursor = conn.cursor()
    sql = "CREATE TABLE %s \
          (systemtime BIGINT NOT NULL, len INT, time1 INT, time2 INT, dummy1 INT, node_id INT, \
          seqno INT, hop INT, dummy2 INT, \
          tem1 INT, hum1 INT, tem2 INT, hum2 INT,\
          sensortime INT)"
    sql2 = "PRAGMA journal_mode=WAL"
    try:
        cursor.execute(sql %(table))
        cursor.execute(sql2)
    except:
        print "Database SHT21 sensors was already created"
    conn.commit()

def createDatabase328Full(conn, table):
    """Function: Read sensor from position i with the contains                 
              (systemtime, acc_x, acc_y, acc_z, ..., sensortime, activity)  
    Input: i is index position of sensor                                    
    Output: sensor data                                                      
    """
    cursor = conn.cursor()
    sql = "CREATE TABLE %s \
          (systemtime BIGINT NOT NULL, node_id INT,  \
          ax1 INT, ay1 INT, az1 INT, gx1 INT, gy1 INT, gz1 INT, mx1 INT, my1 INT, mz1 INT, \
          ax2 INT, ay2 INT, az2 INT, gx2 INT, gy2 INT, gz2 INT, mx2 INT, my2 INT, mz2 INT, \
          ax3 INT, ay3 INT, az3 INT, gx3 INT, gy3 INT, gz3 INT, mx3 INT, my3 INT, mz3 INT, \
          ax4 INT, ay4 INT, az4 INT, gx4 INT, gy4 INT, gz4 INT, mx4 INT, my4 INT, mz4 INT, \
          sensortime INT, activity INT DEFAULT NULL)"
    sql2 = "PRAGMA journal_mode=WAL"
    try:
        cursor.execute(sql %(table))
        cursor.execute(sql2)
    except:
        print "Database was already created"
    conn.commit()

def createDatabase(conn, table):
    """Function: Read sensor from position i with the contains
              (systemtime, acc_x, acc_y, acc_z, ..., sensortime, activity)
    Input: i is index position of sensor                                  
    Output: sensor data                                                    
    """
    cursor = conn.cursor()
    sql = "CREATE TABLE %s \
          (systemtime BIGINT NOT NULL, node_id INT,  \
          ax1 INT, ay1 INT, az1 INT, gx1 INT, gy1 INT, gz1 INT, \
          ax2 INT, ay2 INT, az2 INT, gx2 INT, gy2 INT, gz2 INT, \
          ax3 INT, ay3 INT, az3 INT, gx3 INT, gy3 INT, gz3 INT, \
          ax4 INT, ay4 INT, az4 INT, gx4 INT, gy4 INT, gz4 INT, \
          ax5 INT, ay5 INT, az5 INT, gx5 INT, gy5 INT, gz5 INT, \
          ax6 INT, ay6 INT, az6 INT, gx6 INT, gy6 INT, gz6 INT, \
          ax7 INT, ay7 INT, az7 INT, gx7 INT, gy7 INT, gz7 INT, \
          ax8 INT, ay8 INT, az8 INT, gx8 INT, gy8 INT, gz8 INT, \
          ax9 INT, ay9 INT, az9 INT, gx9 INT, gy9 INT, gz9 INT, \
          ax10 INT, ay10 INT, az10 INT, gx10 INT, gy10 INT, gz10 INT, \
          ax11 INT, ay11 INT, az11 INT, gx11 INT, gy11 INT, gz11 INT, \
          ax12 INT, ay12 INT, az12 INT, gx12 INT, gy12 INT, gz12 INT, \
          ax13 INT, ay13 INT, az13 INT, gx13 INT, gy13 INT, gz13 INT, \
          ax14 INT, ay14 INT, az14 INT, gx14 INT, gy14 INT, gz14 INT, \
          ax15 INT, ay15 INT, az15 INT, gx15 INT, gy15 INT, gz15 INT, \
          ax16 INT, ay16 INT, az16 INT, gx16 INT, gy16 INT, gz16 INT, \
          sensortime INT, activity INT DEFAULT NULL)"
    sql2 = "PRAGMA journal_mode=WAL"
    try:
        cursor.execute(sql %(table))
        cursor.execute(sql2)
    except:
        print "Database was already created"
    conn.commit()
  
def updateSensorDataTem(conn, table, data):
    cursor = conn.cursor()
    sql = '''INSERT INTO ''' + table + ''' \
          (systemtime, len, time1, time2, dummy1, node_id, \
          seqno, hop, dummy2, \
          tem1, hum1, tem2, hum2,\
          sensortime)\
          VALUES (?, ?, ?, ?, ?, ?,\
          ?, ?, ?, \
          ?, ?, ?, ?,\
          ?)'''
    cursor.execute(sql, (data[0], data[1], data[2], data[3], data[4], data[5],\
        data[6], data[7], data[8],\
        data[9], data[10], data[11], data[12],
        data[13]))
    conn.commit()

def updateSensorDataAcc32(conn, table, data):
    cursor = conn.cursor()
    sql = '''INSERT INTO ''' + table + ''' \
          (systemtime, len, time1, time2, dummy1, node_id, \
          seqno, hop, dummy2, \
          ax1, ay1, az1, ax2, ay2, az2, ax3, ay3, az3, ax4, ay4, az4, \
          ax5, ay5, az5, ax6, ay6, az6, ax7, ay7, az7, ax8, ay8, az8, \
          sensortime) \
          VALUES (?, ?, ?, ?, ?, ?,\
          ?, ?, ?, \
          ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,\
          ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,\
          ?)'''
    cursor.execute(sql, (data[0], data[1], data[2], data[3], data[4], data[5],\
        data[6], data[7], data[8],\
        data[9], data[10], data[11], data[12], data[13], data[14],\
        data[15], data[16], data[17], data[18], data[19], data[20],\
        data[21], data[22], data[23], data[24], data[25], data[26],\
        data[27], data[28], data[29], data[30], data[31], data[32],\
        data[33]))
    conn.commit()

def updateSensorDataAcc(conn, table, data):
    cursor = conn.cursor()
    sql = '''INSERT INTO ''' + table + ''' \
          (systemtime, len, time1, time2, dummy1, node_id, \
          seqno, hop, dummy2, \
          ax1, ay1, az1, ax2, ay2, az2, ax3, ay3, az3, ax4, ay4, az4, \
          ax5, ay5, az5, ax6, ay6, az6, ax7, ay7, az7, ax8, ay8, az8, \
          ax9, ay9, az9, ax10, ay10, az10, \
          sensortime) \
          VALUES (?, ?, ?, ?, ?, ?,\
          ?, ?, ?, \
          ?, ?, ?,  ?, ?, ?,  ?, ?, ?,  ?, ?, ?,\
          ?, ?, ?,  ?, ?, ?,  ?, ?, ?,  ?, ?, ?,\
          ?, ?, ?,  ?, ?, ?,\
          ?)'''
    cursor.execute(sql, (data[0], data[1], data[2], data[3], data[4], data[5],\
        data[6], data[7], data[8],\
        data[9], data[10], data[11],   data[12], data[13], data[14],\
        data[15], data[16], data[17],  data[18], data[19], data[20],\
        data[21], data[22], data[23],  data[24], data[25], data[26],\
        data[27], data[28], data[29],  data[30], data[31], data[32],\
        data[33], data[34], data[35],  data[36], data[37], data[38],\
        data[39]))
    conn.commit()

def updateSensorDataEcg16(conn, table, data):
    cursor = conn.cursor()
    sql = '''INSERT INTO ''' + table + ''' \
          (systemtime, len, time1,  time2, state, node_id, \
          seqno, hop, rssi, \
          x1, x2, x3,     x4, x5, x6,     x7, x8, x9,    x10, x11, x12, \
          x13, x14, x15,  x16,\
          mac_stat) \
          VALUES (?, ?, ?,  ?, ?, ?,\
          ?, ?, ?, \
          ?, ?, ?,  ?, ?, ?,  ?, ?, ?,  ?, ?, ?,\
          ?, ?, ?,  ?, \
          ?) '''
    cursor.execute(sql, (data[0], data[1], data[2], data[3], data[4], data[5],\
        data[6], data[7], data[8],\
        data[9], data[10], data[11],  data[12], data[13], data[14],\
        data[15], data[16], data[17], data[18], data[19], data[20],\
        data[21], data[22], data[23], data[24],\
        data[25]))
    conn.commit()

def updateSensorDataEcg(conn, table, data):
    cursor = conn.cursor()
    sql = '''INSERT INTO ''' + table + ''' \
          (systemtime, len, time1,  time2, state, node_id, \
          seqno, hop, rssi, \
          x1, x2, x3,     x4, x5, x6,     x7, x8, x9,    x10, x11, x12, \
          x13, x14, x15,  x16, x17, x18,  x19, x20, x21, x22, x23, x24, \
          x25, x26, x27,  x28, x29, x30,  x31, x32, \
          mac_stat) \
          VALUES (?, ?, ?,  ?, ?, ?,\
          ?, ?, ?, \
          ?, ?, ?,  ?, ?, ?,  ?, ?, ?,  ?, ?, ?,\
          ?, ?, ?,  ?, ?, ?,  ?, ?, ?,  ?, ?, ?,\
          ?, ?, ?,  ?, ?, ?,  ?, ?,\
          ?) '''
    cursor.execute(sql, (data[0], data[1], data[2], data[3], data[4], data[5],\
        data[6], data[7], data[8],\
        data[9], data[10], data[11],  data[12], data[13], data[14],\
        data[15], data[16], data[17], data[18], data[19], data[20],\
        data[21], data[22], data[23], data[24], data[25], data[26],\
        data[27], data[28], data[29], data[30], data[31], data[32],\
        data[33], data[34], data[35], data[36], data[37], data[38],\
        data[39], data[40], data[41]))
    conn.commit()
    
# This method is designed for update sensor data with 32Hz and 8 packets/s
def updateSensorData328(conn, table, data):
    cursor = conn.cursor()
    sql = '''INSERT INTO ''' + table + ''' \
           (systemtime, node_id, \
           ax1, ay1, az1, gx1, gy1, gz1, ax2, ay2, az2, gx2, gy2, gz2, \
           ax3, ay3, az3, gx3, gy3, gz3, ax4, ay4, az4, gx4, gy4, gz4, \
           sensortime)  \
          VALUES (?,?, \
          ?,?,?, ?,?,?, ?,?,?, ?,?,?, ?,?,?, ?,?,?, ?,?,?, ?,?,?, \
          ?)''' 
    cursor.execute(sql, (data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15], data[16], data[17], data[18], data[19], data[20], data[21], data[22], data[23], data[24], data[25], data[26]))
    conn.commit()

# This method is designed for update full sensors data with 32Hz and 8 packets/s
def updateSensorData328Full(conn, table, data):
    cursor = conn.cursor()
    sql = '''INSERT INTO ''' + table + ''' \
           (systemtime, node_id, \
           ax1, ay1, az1, gx1, gy1, gz1, mx1, my1, mz1, ax2, ay2, az2, gx2, gy2, gz2, mx2, my2, mz2,\
           ax3, ay3, az3, gx3, gy3, gz3, mx3, my3, mz3, ax4, ay4, az4, gx4, gy4, gz4, mx4, my4, mz4,\
           sensortime)  \
          VALUES (?,?, \
          ?,?,?, ?,?,?, ?,?,?,\
	  ?,?,?, ?,?,?, ?,?,?,\
	  ?,?,?, ?,?,?, ?,?,?,\
	  ?,?,?, ?,?,?, ?,?,?,\
          ?)''' 
    cursor.execute(sql, (data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], \
	data[9], data[10], data[11], data[12], data[13], data[14], data[15], data[16], \
	data[17], data[18], data[19], data[20], data[21], data[22], data[23], data[24], \
	data[25], data[26], data[27], data[28], data[29], data[30], data[31], data[32], \
	data[33], data[34], data[35], data[36], data[37], data[38]))
    conn.commit()


def updateSensorData(conn, table, data):
    cursor = conn.cursor()
    sql = '''INSERT INTO ''' + table + ''' \
           (systemtime, node_id, \
           ax1, ay1, az1, gx1, gy1, gz1, ax2, ay2, az2, gx2, gy2, gz2, \
           ax3, ay3, az3, gx3, gy3, gz3, ax4, ay4, az4, gx4, gy4, gz4, \
           ax5, ay5, az5, gx5, gy5, gz5, ax6, ay6, az6, gx6, gy6, gz6, \
           ax7, ay7, az7, gx7, gy7, gz7, ax8, ay8, az8, gx8, gy8, gz8, \
           ax9, ay9, az9, gx9, gy9, gz9, ax10, ay10, az10, gx10, gy10, gz10, \
           ax11, ay11, az11, gx11, gy11, gz11, ax12, ay12, az12, gx12, gy12, gz12, \
           ax13, ay13, az13, gx13, gy13, gz13, ax14, ay14, az14, gx14, gy14, gz14, \
           ax15, ay15, az15, gx15, gy15, gz15, ax16, ay16, az16, gx16, gy16, gz16, \
           sensortime)  \
          VALUES (?,?, \
          ?,?,?, ?,?,?, ?,?,?, ?,?,?, ?,?,?, ?,?,?, ?,?,?, ?,?,?, \
          ?,?,?, ?,?,?, ?,?,?, ?,?,?, ?,?,?, ?,?,?, ?,?,?, ?,?,?, \
          ?,?,?, ?,?,?, ?,?,?, ?,?,?, ?,?,?, ?,?,?, ?,?,?, ?,?,?, \
          ?,?,?, ?,?,?, ?,?,?, ?,?,?, ?,?,?, ?,?,?, ?,?,?, ?,?,?, \
          ?)''' 
    cursor.execute(sql, (data[0], data[1],\
        data[2], data[3], data[4],  data[5], data[6], data[7],  data[8], data[9], data[10],  data[11], data[12], data[13],\
        data[14], data[15], data[16],  data[17], data[18], data[19],  data[20], data[21], data[22],  data[23], data[24], data[25],\
        data[26], data[27], data[28],  data[29], data[30], data[31],  data[32], data[33], data[34],  data[35], data[36], data[37],\
        data[38], data[39], data[40],  data[41], data[42], data[43],  data[44], data[45],  data[46], data[47], data[48], data[49],\
        data[50], data[51], data[52], data[53], data[54], data[55], data[56], data[57], data[58], data[59], data[60], data[61],\
        data[62], data[63], data[64], data[65], data[66], data[67], data[68], data[69], data[70], data[71], data[72], data[73],\
        data[74], data[75], data[76], data[77], data[78], data[79], data[80], data[81], data[82], data[83], data[84], data[85],\
        data[86], data[87], data[88], data[89], data[90], data[91], data[92], data[93], data[94], data[95], data[96], data[97],\
        data[98]))
    conn.commit()


def readSensorDataCLF328(cursor, table, i):
    """Function: Read sensor from position i with the contains                 
              (systemtime, acc_x, acc_y, acc_z, ..., sensortime, activity)  
    Input: i is index position of sensor                                    
    Output: sensor data                                                      
    """
    sql="SELECT systemtime, \
         ax1, ay1, az1, gx1, gy1, gz1, ax2, ay2, az2, gx2, gy2, gz2, \
         ax3, ay3, az3, gx3, gy3, gz3, ax4, ay4, az4, gx4, gy4, gz4, \
         sensortime, activity \
         FROM %s WHERE (node_id = %s) AND (activity IS NOT NULL)"
    cursor.execute(sql %(table, ID[i]))
    temp = cursor.fetchall()
    return np.asarray(temp)

def readSensorDataCLF328Full(cursor, table, i):
    """
    Function: Read sensor from position i with the contains                
              (systemtime, acc_x, acc_y, acc_z, ..., sensortime, activity) 
    Input: i is index position of sensor                                   
    Output: sensor data                                                     
    """
    sql="SELECT systemtime, \
         ax1, ay1, az1, gx1, gy1, gz1, mx1, my1, mz1,\
         ax2, ay2, az2, gx2, gy2, gz2, mx2, my2, mz2, \
         ax3, ay3, az3, gx3, gy3, gz3, mx3, my3, mz3, \
         ax4, ay4, az4, gx4, gy4, gz4, mx4, my4, mz4, \
         sensortime, activity \
         FROM %s WHERE (node_id = %s) AND (activity IS NOT NULL)"
    cursor.execute(sql %(table, ID[i]))
    temp = cursor.fetchall()
    return np.asarray(temp)

def readSensorDataCLF(cursor, table, i):
    """Function: Read sensor from position i with the contains
              (systemtime, acc_x, acc_y, acc_z, ..., sensortime, activity)
    Input: i is index position of sensor                                  
    Output: sensor data                                                    
    """
    sql="SELECT systemtime, \
         ax1, ay1, az1, gx1, gy1, gz1, ax2, ay2, az2, gx2, gy2, gz2, \
         ax3, ay3, az3, gx3, gy3, gz3, ax4, ay4, az4, gx4, gy4, gz4, \
         ax5, ay5, az5, gx5, gy5, gz5, ax6, ay6, az6, gx6, gy6, gz6, \
         ax7, ay7, az7, gx7, gy7, gz7, ax8, ay8, az8, gx8, gy8, gz8, \
         ax9, ay9, az9, gx9, gy9, gz9, ax10, ay10, az10, gx10, gy10, gz10, \
         ax11, ay11, az11, gx11, gy11, gz11, ax12, ay12, az12, gx12, gy12, gz12, \
         ax13, ay13, az13, gx13, gy13, gz13, ax14, ay14, az14, gx14, gy14, gz14, \
         ax15, ay15, az15, gx15, gy15, gz15, ax16, ay16, az16, gx16, gy16, gz16, \
         sensortime, activity \
         FROM %s WHERE (node_id = %s) AND (activity IS NOT NULL)"
    cursor.execute(sql %(table, ID[i]))
    temp = cursor.fetchall()
    return np.asarray(temp)

def readSensorData(cursor, table, i, st):
    """Function: Read sensor from position i with the contains              
              (systemtime, acc_x, acc_y, acc_z, ..., sensortime, activity)  
    Input: i is index position of sensor and sending timestamp              
    Output: sensor data                                                     
    """
    sql="SELECT systemtime, \
         ax1, ay1, az1, gx1, gy1, gz1, ax2, ay2, az2, gx2, gy2, gz2, \
         ax3, ay3, az3, gx3, gy3, gz3, ax4, ay4, az4, gx4, gy4, gz4, \
         ax5, ay5, az5, gx5, gy5, gz5, ax6, ay6, az6, gx6, gy6, gz6, \
         ax7, ay7, az7, gx7, gy7, gz7, ax8, ay8, az8, gx8, gy8, gz8, \
         ax9, ay9, az9, gx9, gy9, gz9, ax10, ay10, az10, gx10, gy10, gz10, \
         ax11, ay11, az11, gx11, gy11, gz11, ax12, ay12, az12, gx12, gy12, gz12, \
         ax13, ay13, az13, gx13, gy13, gz13, ax14, ay14, az14, gx14, gy14, gz14, \
         ax15, ay15, az15, gx15, gy15, gz15, ax16, ay16, az16, gx16, gy16, gz16, \
         sensortime \
         FROM %s WHERE (node_id = %s) AND (systemtime > %s) \
		 LIMIT 4"
         # AND (activity IS NULL) LIMIT 4"
    cursor.execute(sql %(table, ID[i], int(st)))
    temp = cursor.fetchall()
    return np.asarray(temp)

def readSensorData2(cursor, table, i, t):
    sql="SELECT systemtime, \
         ax1, ay1, az1, gx1, gy1, gz1, ax2, ay2, az2, gx2, gy2, gz2, \
         ax3, ay3, az3, gx3, gy3, gz3, ax4, ay4, az4, gx4, gy4, gz4, \
         ax5, ay5, az5, gx5, gy5, gz5, ax6, ay6, az6, gx6, gy6, gz6, \
         ax7, ay7, az7, gx7, gy7, gz7, ax8, ay8, az8, gx8, gy8, gz8, \
         ax9, ay9, az9, gx9, gy9, gz9, ax10, ay10, az10, gx10, gy10, gz10, \
         ax11, ay11, az11, gx11, gy11, gz11, ax12, ay12, az12, gx12, gy12, gz12, \
         ax13, ay13, az13, gx13, gy13, gz13, ax14, ay14, az14, gx14, gy14, gz14, \
         ax15, ay15, az15, gx15, gy15, gz15, ax16, ay16, az16, gx16, gy16, gz16, \
         sensortime \
         FROM %s WHERE (node_id = %s) \
         AND (sensortime = %s OR sensortime = %s) \
		 LIMIT 4"
		 # AND (activity IS NULL) LIMIT 4"
    cursor.execute(sql %(table, ID[i], int(t), int(t)+1))
    temp = cursor.fetchall()
    return np.asarray(temp)

def readSensorDataEcg(cursor, table, starttime, endtime):
    sql = '''SELECT systemtime, len, time1,  time2, state, node_id, \
          seqno, hop, rssi, \
          x1, x2, x3,     x4, x5, x6,     x7, x8, x9,    x10, x11, x12, \
          x13, x14, x15,  x16, x17, x18,  x19, x20, x21, x22, x23, x24, \
          x25, x26, x27,  x28, x29, x30,  x31, x32, \
          mac_stat  \
          FROM %s WHERE (systemtime > %s AND systemtime < %s)'''
    cursor.execute(sql %(table, int(starttime), int(endtime)))
    temp = cursor.fetchall()
    return np.asarray(temp)

def readSensorDataAcc(cursor, table, starttime, endtime):
    sql = '''SELECT systemtime, len, time1, time2, dummy1, node_id, \
          seqno, hop, dummy2, \
          ax1, ay1, az1, ax2, ay2, az2, ax3, ay3, az3, ax4, ay4, az4, \
          ax5, ay5, az5, ax6, ay6, az6, ax7, ay7, az7, ax8, ay8, az8, \
          ax9, ay9, az9, ax10, ay10, az10, \
          sensortime \
          FROM %s WHERE (systemtime > %s AND systemtime < %s)'''
    cursor.execute(sql %(table, int(starttime), int(endtime)))
    temp = cursor.fetchall()
    return np.asarray(temp)

def readSensorDataAccFull(cursor, table):
    sql = '''SELECT systemtime, len, time1, time2, dummy1, node_id, \
          seqno, hop, dummy2, \
          ax1, ay1, az1, ax2, ay2, az2, ax3, ay3, az3, ax4, ay4, az4, \
          ax5, ay5, az5, ax6, ay6, az6, ax7, ay7, az7, ax8, ay8, az8, \
          ax9, ay9, az9, ax10, ay10, az10, \
          sensortime, activity \
          FROM %s WHERE activity IS NOT NULL'''
    cursor.execute(sql %(table))
    temp = cursor.fetchall()
    return np.asarray(temp)

def readSensorDataAccTrain(cursor, table):
    sql = '''SELECT 
          ax1, ay1, az1, ax2, ay2, az2, ax3, ay3, az3, ax4, ay4, az4, \
          ax5, ay5, az5, ax6, ay6, az6, ax7, ay7, az7, ax8, ay8, az8, \
          ax9, ay9, az9, ax10, ay10, az10, \
          activity \
          FROM %s WHERE activity IS NOT NULL'''
    cursor.execute(sql %(table))
    temp = cursor.fetchall()
    return np.asarray(temp)

def readSensorDataTem(cursor, table, starttime, endtime):
    sql = '''SELECT systemtime, len, time1, time2, dummy1, node_id, \
          seqno, hop, dummy2, \
          tem1, hum1, tem2, hum2,\
          sensortime\
          FROM %s WHERE (systemtime > %s AND systemtime < %s)'''
    cursor.execute(sql %(table, int(starttime), int(endtime)))
    temp = cursor.fetchall()
    return np.asarray(temp)

def readStartTime(cursor, table, i):
    sql = "SELECT systemtime, sensortime FROM %s WHERE node_id = %s AND activity IS NULL LIMIT 1"
    cursor.execute(sql % (table, ID[i]))
    temp = cursor.fetchone()
    return temp[0], temp[1]      

def readStartTime2(cursor, table, i, systemtime):
    sql = "SELECT systemtime, sensortime FROM %s WHERE node_id = %s AND systemtime > %s AND\
           activity IS NULL LIMIT 1"
    cursor.execute(sql % (table, ID[i], systemtime))
    temp = cursor.fetchone()
    return temp[0], temp[1]      

def readLastTime(cursor, table, i):
    sql = "SELECT systemtime, sensortime FROM %s WHERE node_id = %s AND activity IS NULL \
           ORDER BY systemtime DESC LIMIT 1"
    cursor.execute(sql % (table, ID[i]))
    temp = cursor.fetchone()
    return temp[0], temp[1]

def readLastTime2(cursor, table, i, systemtime):
    sql = "SELECT systemtime, sensortime FROM %s WHERE node_id = %s AND systemtime > %s AND\
          activity IS NULL ORDER BY systemtime DESC LIMIT 1"
    cursor.execute(sql % (table, ID[i], systemtime))
    temp = cursor.fetchone()
    return temp[0], temp[1]
