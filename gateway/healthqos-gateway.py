#!/usr/bin/python

from __future__ import division
import socket
import binascii
import time as stime
import pygame
import logging
import threading
import sys, getopt
import sqlite3
import Queue
import serial

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from pyqtgraph.ptime import time
from pyqtgraph import ViewBox
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn import tree
from sklearn import svm

from harCommon import *
# Netperf library for network performance analysis
from netperf import *
from visualizeCommon import *

__author__ = "Ngo Van Mao"
__copyright__ = "Copyright 2017, Singapore University of Technology and Design (SUTD)"
__credits__ = ["Ngo Van Mao"]
__license__ = "GNU GPLv3.0"
__version__ = "2.0.0"
__maintainer__ = "Ngo Van Mao"
__email__ = "vanmao_ngo@mymail.sutd.edu.sg"
__status__ = "Development"

"""
================================================================================
                              GLOBAL VARIABLES
================================================================================
"""
SERVER_IP = "fd00::1"
slip_radio_ip = "fd00::9999"
SERVER_PORT= 5688
Width=512; Height = 512 
white = (255,255,255)
black = (0,0,0)
TRAIN_DIR = 'trainedModel/basic_'
SVM_STORINGMODEL = TRAIN_DIR + 'SVM_SensorTag.pkl'
DCT_STORINGMODEL = TRAIN_DIR + 'DCT_SensorTag.pkl'
RF_STORINGMODEL = TRAIN_DIR + 'RF_SensorTag.pkl'
TRAINED_MODEL = RF_STORINGMODEL
SCALER_STORINGMODEL = TRAIN_DIR + 'SCALER_SensorTag.pkl'
START_TIME = stime.strftime("%H:%M:%S")
SERIALPORT = "/dev/ttyUSB0"
use_serial = False

DATABASE = "ServerDatabase.db"
TABLE = "ServerTable" # Sit, Standing, Run, walk, 
PREDICT = False
useTCP = False
DEBUG = False #True for extra print statements regarding raw input

ecg_pdr = 0.00
tem_pdr = 0.00
acc_pdr = 0.00
ecg_thrput = 0.00
tem_thrput = 0.00
acc_thrput = 0.00
ecg_jitter = 0
acc_jitter = 0
tem_jitter = 0

sensors = [] #List of sensors updated in the last update cycle
first_acc = True #True if no datapoints have been read in yet (ie current data 
             #point is 'first') set to false otherwise

is_triggerring = 0
trigger_command = ""
packetWindow = 6

"""
================================================================================
                            END GLOBAL VARIABLES
================================================================================
"""
"""
================================================================================
                           GRAPH PLOT START
================================================================================
"""
app = QtGui.QApplication([])
w = QtGui.QWidget()
pg.setConfigOption('background', 'w')

pACC = pg.PlotWidget(axisItems={'bottom': TimeAxisItem(orientation='bottom')})
pTEM = pg.PlotWidget(axisItems={'bottom': TimeAxisItem(orientation='bottom')})
pHUM = pg.PlotWidget(axisItems={'bottom': TimeAxisItem(orientation='bottom')})
pECG = pg.PlotWidget(axisItems={'bottom': TimeAxisItem(orientation='bottom')})
pThrputEcg = pg.PlotWidget(axisItems={'bottom': TimeAxisItem(orientation='bottom')})
pThrputAcc = pg.PlotWidget(axisItems={'bottom': TimeAxisItem(orientation='bottom')})
pThrputTem = pg.PlotWidget(axisItems={'bottom': TimeAxisItem(orientation='bottom')})
pPDREcg = pg.PlotWidget(axisItems={'bottom': TimeAxisItem(orientation='bottom')})
pPDRAcc = pg.PlotWidget(axisItems={'bottom': TimeAxisItem(orientation='bottom')})
pPDRTem = pg.PlotWidget(axisItems={'bottom': TimeAxisItem(orientation='bottom')})


pACC.setTitle('Accelerometer')
pTEM.setTitle('Temperature & Humidity')
pECG.setTitle('ECG')
pThrputEcg.setTitle('Throughput Ecg (bits/s)')
pThrputAcc.setTitle('Throughput Acc (bits/s)')
pThrputTem.setTitle('Throughput Tem (bits/s)')
pPDREcg.setTitle('PDR Ecg (%)')
pPDRAcc.setTitle('PDR Acc (%)')
pPDRTem.setTitle('PDR Tem (%)')
pACC.setYRange(-FULL_ACC, FULL_ACC, padding = 0)
pTEM.setYRange(0, 100, padding = 0)
pECG.setYRange(0, 1000, padding = 0)

pACC.enableAutoRange(ViewBox.XAxis)
pTEM.enableAutoRange(ViewBox.XAxis)
pECG.enableAutoRange(ViewBox.XAxis)
pACC.enableAutoRange(ViewBox.YAxis)
pTEM.enableAutoRange(ViewBox.YAxis)
pECG.enableAutoRange(ViewBox.YAxis)
pThrputEcg.enableAutoRange(ViewBox.XAxis)
pThrputEcg.enableAutoRange(ViewBox.YAxis)
pThrputAcc.enableAutoRange(ViewBox.XAxis)
pThrputAcc.enableAutoRange(ViewBox.YAxis)
pThrputTem.enableAutoRange(ViewBox.XAxis)
pThrputTem.enableAutoRange(ViewBox.YAxis)
pPDREcg.enableAutoRange(ViewBox.XAxis)
pPDREcg.enableAutoRange(ViewBox.YAxis)
pPDRAcc.enableAutoRange(ViewBox.XAxis)
pPDRAcc.enableAutoRange(ViewBox.YAxis)
pPDRTem.enableAutoRange(ViewBox.XAxis)
pPDRTem.enableAutoRange(ViewBox.YAxis)

pACC.addLegend()
pTEM.addLegend()
pECG.addLegend()
pThrputEcg.addLegend()
pThrputAcc.addLegend()
pThrputTem.addLegend()
pPDREcg.addLegend()
pPDRAcc.addLegend()
pPDRTem.addLegend()

curveAx = pACC.plot(pen='r', name="Accx")
curveAy = pACC.plot(pen='g', name="Accy")
curveAz = pACC.plot(pen='b', name="Accz")
curveTem = pTEM.plot(pen='r', name="Temperature")
curveHum = pTEM.plot(pen='g', name="Humidity")
curveEcg = pECG.plot(pen='y', name="ECG")
curveThrputEcg = pThrputEcg.plot(pen = 'r', name='Thrput_ECG')
curveThrputAcc = pThrputAcc.plot(pen = 'g', name='Thrput_ACC')
curveThrputTem = pThrputTem.plot(pen = 'b', name='Thrput_TEM')
curvePdrEcg = pPDREcg.plot(pen = 'r', name='PDR_ECG')
curvePrrEcg = pPDREcg.plot(pen = 'y', name='PRR_ECG')
curvePdrAcc = pPDRAcc.plot(pen = 'g', name='PDR_ACC')
curvePdrTem = pPDRTem.plot(pen = 'b', name='PDR_TEM')


w.show()

MAX_LENGTH = 1000  # determines length of data taking session (in data points)
# create empty variable of length of test
ax = [0] * MAX_LENGTH; ay = [0] * MAX_LENGTH; az = [0] * MAX_LENGTH 
tem = [0] * MAX_LENGTH; hum = [0] * MAX_LENGTH;
ecg = [0] * MAX_LENGTH
thrput_ecg = [0] * MAX_LENGTH
thrput_acc = [0] * MAX_LENGTH; thrput_tem = [0] * MAX_LENGTH; 

pdr_ecg = [0] * MAX_LENGTH; prr_ecg = [0] * MAX_LENGTH
pdr_acc = [0] * MAX_LENGTH; pdr_tem = [0] * MAX_LENGTH

acc_ts = [0] * MAX_LENGTH; tem_ts = [0] * MAX_LENGTH; ecg_ts = [0] * MAX_LENGTH
ecg_net_ts = [0] * MAX_LENGTH; acc_net_ts = [0] * MAX_LENGTH; tem_net_ts = [0] * MAX_LENGTH

"""
================================================================================
                           GRAPH PLOT START
================================================================================
"""

def updateSensor():
    #ms = stime.time() * 1000
    global sensors
    for sensor in sensors:
        if sensor == MAC_TEM:
            try:
                curveTem.setData(tem_ts, tem)
                curveHum.setData(tem_ts, hum)
                curveThrputTem.setData(tem_net_ts, thrput_tem)
                curvePdrTem.setData(tem_net_ts, pdr_tem)
            except:
                #print sys.exec_info()[0]
                pass
        if sensor == MAC_ACC:
            try:
                curveAx.setData(acc_ts, ax)
                curveAy.setData(acc_ts, ay)
                curveAz.setData(acc_ts, az)
                curveThrputAcc.setData(acc_net_ts, thrput_acc)
                curvePdrAcc.setData(acc_net_ts, pdr_acc)
            except:
                #print sys.exec_info()[0]
                pass
        if sensor == MAC_ECG:
            try:
                curveEcg.setData(ecg_ts, ecg)
                #print thrput
                curveThrputEcg.setData(ecg_net_ts, thrput_ecg)
                curvePdrEcg.setData(ecg_net_ts, pdr_ecg)
                curvePrrEcg.setData(ecg_net_ts, prr_ecg)
            except:
                #print sys.exec_info()[0]
                pass
    sensors = []

def parse_args(argv):
    """Function: Reads in command line arguments and makes alterations to global
              variables
    Input:    Command line arguments excluding the script name
    Mutates:  Changes values of global variables
    Output:   None
    """
    global DATABASE, TABLE, PREDICT, useTCP  
    global DEBUG, SERVER_IP, SERVER_PORT, TRAINED_MODEL, SERIALPORT, use_serial
    global TRAIN_DIR,SVM_STORINGMODEL,DCT_STORINGMODEL,RF_STORINGMODEL,SCALER_STORINGMODEL
    print 'healthqos-server BSN, version ' + __version__
    DIR = 'trainedModel/'
    NAME = 'basic_'
    try:
       opts, args = getopt.getopt(argv,"h:TDra:d:t:m:p:",["model=","addr=","database=","table=",
                       "model=","ACC=","all","port="])
    except getopt.GetoptError:
       print './healthqos-server.py -h:a:d:t:i:T:m:r: [-d <database>] [-t <table>][-m <model>]'
       print "                      [--ACC<ACCID>][--TEM<TEMID>][--dir<dir>]"
       print "                      [--model<modelname>]"
       print "    -a : open server on address:port. Default is localhost:7788"
       print "    -d : data base name. Default is testDatabase.db"
       print "    -D : Debug printout"
       print "    -r : run prediction mode"
       print "    -m : Trained model"
       print "    -T : Run TCP server. Default, collector-server runs on UDP"
       print "    -t : input table name"
       print "    -p : serial port. Default is /dev/ttyUSB0"
       print "--dir  : Directory of training model"
       print "--model: Model name, within the directory"
       print "Ex: ./collector-server.py -ACC 14470"
       print "   Run server with default values, just change the ACCID to 14470"
       sys.exit(2)
    for opt, arg in opts:
       if opt in ('-a', '--addr'):
          dest = arg
          SERVER_IP = dest.rpartition(':')[0]
          SERVER_PORT = dest.rpartition(':')[-1]
          SERVER_PORT = int(SERVER_PORT)
       if opt == '-D':
          DEBUG = True
          print "Run DEBUG mode"
       if opt == '-T':
          useTCP = True
          print 'Run TCP'
       if opt == '-r':
          PREDICT = True
          print 'Run PREDICT mode'
       if opt in ('-p', "--port"):
          use_serial = True
          SERIALPORT = arg
       if opt in ("-d", "--database"):
          DATABASE = arg
       if opt in ("-t", "--table"):
          TABLE = arg
       if opt in ("-m", "--model"):
          TRAINED_MODEL = arg
       if opt in ("--ACC",):
          ACCID = arg
          ID[1]=ACCID
          print "ACCID = ", ACCID
       if opt in ("--TEM",):
          TEMID = arg
          ID[4]=TEMID
          print "TEMID = ", TEMID
       if opt in ("--model",):
          NAME = '{}_'.format(arg)
       if opt in ("--dir","--directory"):
          DIR = '{}/'.format(arg)
    TRAIN_DIR = DIR + NAME
    DATABASE = DB_DIR + DATABASE
    SVM_STORINGMODEL = TRAIN_DIR + 'SVM_SensorTag.pkl'
    DCT_STORINGMODEL = TRAIN_DIR + 'DCT_SensorTag.pkl'
    RF_STORINGMODEL = TRAIN_DIR + 'RF_SensorTag.pkl'
    TRAINED_MODEL = RF_STORINGMODEL
    SCALER_STORINGMODEL = TRAIN_DIR + 'SCALER_SensorTag.pkl'

def showImage(activity):
    """
     Function: Display the activity image                                    
     Input: activity index                                                   
    """
    activityImg = pygame.image.load(img[activity])
    #label = myFont.render(img[activity], 1, (255,0,0))
    ecg_pdr_value = netperfFont.render(str(format(ecg_pdr, '.2f')+"  "+format(ecg_thrput, '.2f')+"  "+format(ecg_jitter)), 1, (255,255,255))
    acc_pdr_value = netperfFont.render(str(format(acc_pdr, '.2f')+"  "+format(acc_thrput, '.2f')+"  "+format(acc_jitter)), 1, (255,255,255))
    tem_pdr_value = netperfFont.render(str(format(tem_pdr, '.2f')+"  "+format(tem_thrput, '.2f')+"  "+format(tem_jitter)), 1, (255,255,255))
    timeLabel = stepFont.render(START_TIME,1,white)
    predictionString = img[activity][11:-4]
    predictionString = predictionString[0:1].upper() + predictionString[1:]
    predictionString = " " * (16-len(predictionString)) + predictionString
    predictionLabel = stepFont.render(predictionString,1,white)
    x = Width * 0
    y = Height * 0
    gameDisplay.fill(white)
    gameDisplay.blit(activityImg, (x,y))
    
    networkPerformance.fill(black)
    networkPerformance.blit(startLabel, (30,0))
    networkPerformance.blit(timeLabel,(50,30))
    networkPerformance.blit(ecg_pdr_label,(30,70))
    networkPerformance.blit(ecg_pdr_value, (65,100))
    networkPerformance.blit(acc_pdr_label,(30,140))
    networkPerformance.blit(acc_pdr_value, (65,170))
    networkPerformance.blit(tem_pdr_label,(30,210))
    networkPerformance.blit(tem_pdr_value, (65,240))
    networkPerformance.blit(activityLabel, (55,280))
    networkPerformance.blit(predictionLabel, (2,310))
    
    pygame.display.update()
    
lastPrediction = len(img)-1 

count_running = 0
def predictionAcc(accArr):
    global lastPrediction, is_triggerring, trigger_command, count_running 
    if len(accArr) > 0:
        # round up raw data upto 16 rows
        if len(accArr) < packetWindow:
            print 'need to round ACC len = ', len(accArr)
        #print 'Converting raw data '
        ACC_fe = extract_feature_acc(accArr, 333)
        ACC_fe_s = ACC_fe.reshape(1, len(ACC_fe))
        X = ACC_fe_s
        X = scaler.transform(X)
        predicted =  clf.predict(X)
        lastPrediction = int(predicted)
        # if predicted value is running. so trigger extending schedule in wsn network.
        if lastPrediction == 1:
            count_running += 1
            print 'running '
            if count_running == 2:
                print 'trigger '
                is_triggerring = 1
                trigger_command = "tg 4 2 120"
                count_running = 0
        #print ("Predicted = ", lastPrediction)
        #writeActivity(TABLE, predicted, 2, t12)
    else:
        print 'no available input sensor data'


class displayDataThread(threading.Thread):
    def __init__(self, data_q):
        super(displayDataThread, self).__init__()
        self.data_q = data_q
        self.stopRequest = threading.Event()
        self.first_tem = True
        self.first_acc = True
        self.first_ecg = True
        self.ecg_netperf = NetPerf(MAC_ECG)
        self.tem_netperf = NetPerf(MAC_TEM)
        self.acc_netperf = NetPerf(MAC_ACC)
        self.preAcc = []

    def parserDataSerial(data):
        # Data format: len, node_id, sensors..., sensortime
        inputData = [val for val in data.split()]
        dataLength = len(inputData)
        if dataLength != (int(inputData[1]) + 1): 
            # return anything with wrong received packet 
            print 'dataLength = ', dataLength, 'inputData = ', inputData
            return 0, inputData, inputData
        #print inputData
        try:
            nodeId = int(inputData[5])
            outData = np.int16(inputData[9:])
        except:
            nodeId = 0
            outData = np.int16([])
            pass
        return nodeId, outData, inputData 
    

    def run(self):
        print 'run Display data thread'
        # The following variables are used for displaying in gameplot
        # instantaneous values.
        global ecg_pdr, ecg_thrput, ecg_jitter
        global acc_pdr, acc_thrput, acc_jitter
        global tem_pdr, tem_thrput, tem_jitter
        # The following global variables are used for real time display.
        global thrput_acc, thrput_tem, thrput_ecg
        global pdr_ecg, pdr_acc, pdr_tem, prr_ecg
        global acc_net_ts, tem_net_ts, ecg_net_ts 
        global sensors
        global acc_ts, ax, ay, az, tem, hum, tem_ts, ecg_ts, ecg 
        flagTotal = 0; count_acc = 0  
        
        while not self.stopRequest.isSet():
            try:
                if use_serial:
                    data = self.data_q.get(True, 0)
                    mac_addr, outData, inputData = self.parserDataSerial(data)
                    data_length = len(outData)
                    ts = int(data.split(' ')[0])
                else:
                    mac_addr, data = self.data_q.get(True, 0)
                    ts = data[0]
                    data_length = data[1] 
                    inputData = data
                    outData = np.asarray(data[9:])

                if data_length < 5:
                    print 'ERROR PACKET data_length = ', data_length, 'outData = ', outData
                else:    
                    sensors.append(mac_addr)
                    # reshape outData to array 1x1 [[xxx]],
                    tempOutData = outData.reshape(1, len(outData))
                    if mac_addr == MAC_ACC:
                        self.acc_netperf.updateNetPerf(inputData)
                        if self.first_acc:
                            tp_ts = self.acc_netperf.systemtime[0]
                            count = MAX_LENGTH
                            c_ts = ts
                            while(count > 0):
                                count -= 1
                                c_ts -= MAX_LENGTH/SAMPLING_RATE_ACC # sampling rate
                                acc_ts[count] = c_ts
                                tp_ts -= MAX_LENGTH
                                acc_net_ts[count] = tp_ts
                            self.first_acc = False
                        else:
                            convertedData = convertDataAcc(tempOutData, 3)
                            # Appending vertically outData to preLH with maxSize is 3 Windows Size
                            self.preAcc = vStackArrays(self.preAcc, convertedData, 3*packetWindow)
                            count_acc += 1
                            if count_acc == packetWindow: # 2s ~ 6 packets
                                flagTotal = 1 
                                count_acc = 0
                            for i in range(len(convertedData)):
                                ax.append(convertedData[i][0])
                                ay.append(convertedData[i][1])
                                az.append(convertedData[i][2])
                                acc_ts.append(ts + i*1000/SAMPLING_RATE_ACC)
                                del ax[0]; del ay[0]; del az[0]; del acc_ts[0]
                            pdr_acc.append(self.acc_netperf.pdr)
                            #jitter_acc.append(self.acc_netperf.jitter[-1])
                            thrput_acc.append(self.acc_netperf.throughput)
                            acc_net_ts.append(self.acc_netperf.systemtime[-1])
                            del pdr_acc[0]; del acc_net_ts[0]; del thrput_acc[0]
                            acc_thrput = self.acc_netperf.throughput
                            acc_jitter = self.acc_netperf.jitter[-1]
                            acc_pdr = self.acc_netperf.pdr
                    if mac_addr == MAC_TEM:
                        self.tem_netperf.updateNetPerf(inputData)
                        if self.first_tem:
                            count = MAX_LENGTH
                            tp_ts = self.tem_netperf.systemtime[0]
                            c_ts = ts
                            while(count > 0):
                                count -= 1
                                c_ts -= MAX_LENGTH / SAMPLING_RATE_TEM 
                                tp_ts -= MAX_LENGTH
                                tem_ts[count] = c_ts
                                tem_net_ts[count] = tp_ts
                            self.first_tem = False
                        else:
                            convertedData = convertDataTem(tempOutData, 2)
                            for i in range(len(convertedData)):
                                tem.append(convertedData[i][0])
                                hum.append(convertedData[i][1])
                                tem_ts.append(ts + i*1000/SAMPLING_RATE_TEM)
                                del tem[0]; del hum[0]; del tem_ts[0]
                            tem_thrput = self.tem_netperf.throughput
                            tem_jitter = self.tem_netperf.jitter[-1]
                            tem_pdr = self.tem_netperf.pdr
                            pdr_tem.append(self.tem_netperf.pdr)
                            thrput_tem.append(self.tem_netperf.throughput)
                            tem_net_ts.append(self.tem_netperf.systemtime[-1])
                            del pdr_tem[0]; del tem_net_ts[0]; del thrput_tem[0];
                    if mac_addr == MAC_ECG:
                        self.ecg_netperf.updateNetPerf(inputData)
                        if self.first_ecg:
                            count = MAX_LENGTH
                            c_ts = ts
                            tp_ts = self.ecg_netperf.systemtime[0]
                            while(count > 0):
                                count -= 1
                                c_ts -= MAX_LENGTH / SAMPLING_RATE_ECG
                                tp_ts -= MAX_LENGTH
                                ecg_ts[count] = c_ts
                                ecg_net_ts[count] = tp_ts
                            self.first_ecg = False
                        else:
                            convertedData = convertDataEcg(tempOutData, 1)
                            for i in range(len(convertedData)):
                                ecg.append(convertedData[i][0])
                                ecg_ts.append(ts + i*1000/SAMPLING_RATE_ECG)
                                del ecg[0]; del ecg_ts[0]
                            ecg_thrput = self.ecg_netperf.throughput
                            ecg_jitter = self.ecg_netperf.jitter[-1]
                            ecg_pdr = self.ecg_netperf.pdr
                            thrput_ecg.append(ecg_thrput) 
                            pdr_ecg.append(ecg_pdr)
                            prr_ecg.append(self.ecg_netperf.getPrr())
                            ecg_net_ts.append(self.ecg_netperf.systemtime[-1])
                            #print 'jitter = ', self.ecg_netperf.jitter[-1]
                            del thrput_ecg[0]; del pdr_ecg[0]; del prr_ecg[0]; del ecg_net_ts[0];
                #TODO generalize this function
                if flagTotal and PREDICT:
                    ACCarr = self.preAcc[(len(self.preAcc) - packetWindow):]
                    predictionAcc(ACCarr) 
                    flagTotal = 0
                showImage(lastPrediction)
            except Queue.Empty:
                continue
                
    def join(self, timeout=None):
        self.stopRequest.set()
        super(displayDataThread, self).join(timeout)
        

"========================================================================="
class udpServerThread(threading.Thread):
    def __init__(self, database, table, data_q):
        threading.Thread.__init__(self)
        self._running = True
        self.sock = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        #self.sock.bind((socket.gethostname(), SERVER_PORT))
        self.sock.bind((SERVER_IP, SERVER_PORT))
        self.data_q = data_q
        self.database = database
        self.table_tem = table + "Tem"
        self.table_acc = table + "Acc"
        self.table_ecg = table + "Ecg"
    
    def parseUdpBinaryData(self, mac_addr, data):
        """sensor data:
        index  0         , 1  , 2    , 3    , 4     , 5      , 6    , 7  , 8
        data =[systemtime, len, time1, time2, state, node_id, seqno, hop, rssi,
               ax1, ay1, az1, ax2, ay2, az2, ax3, ay3, az3, ax4, ay4, az4, \
               ax5, ay5, az5, ax6, ay6, az6, ax7, ay7, az7, ax8, ay8, az8, \
               mac_stat] \
        """
        systemTime = int(round(time.time() * 1000))
        p = binascii.hexlify(data)
        outData = []
        outData.append(systemTime) # systemtime
        outData.append(len(p)) # len
        outData.append(0) # time1
        outData.append(0) # time2
        outData.append(int(p[2:4], 16)) # state
        outData.append(mac_addr) # node_id
        outData.append(int(p[0:2], 16)) # seqno
        outData.append(0) # hop
        outData.append(0) # rssi
        for i in range(4, len(p), 4):
            temp = p[i+2:i+4] + p[i:i+2]
            outData.append(int(temp, 16))
        return outData

    def run(self):
        print 'run UDP thread'
        global ecg_pdr, acc_pdr, tem_pdr, is_triggerring, trigger_command 
        if not os.path.exists(DB_DIR):
            os.makedirs(DB_DIR)
        conn = sqlite3.connect(self.database)
        createDatabaseTem(conn, self.table_tem);
        createDatabaseAcc(conn, self.table_acc);
        createDatabaseEcg(conn, self.table_ecg);
        while self._running:
            data, addr = self.sock.recvfrom(1024) # buffer size is 1024 bytes
            #print "received message from ", addr, "with ", binascii.hexlify(data)
            MAC_addr = int(addr[0][-4:], 16) # 16 is for convert hex to decimal
            outData = self.parseUdpBinaryData(MAC_addr, data)
            #print "outData = ", outData
            #start = stime.time()
            if MAC_addr == MAC_TEM:
                updateSensorDataTem(conn, self.table_tem, outData)
                self.data_q.put((MAC_addr, outData))
            elif MAC_addr == MAC_ACC:
                updateSensorDataAcc(conn, self.table_acc, outData)
                self.data_q.put((MAC_addr, outData))
                #self.data_q.put(outData)
            elif MAC_addr == MAC_ECG:
                updateSensorDataEcg(conn, self.table_ecg, outData)
                self.data_q.put((MAC_addr, outData))
            else:
                print "Error" , outData
            #end = stime.time()
            #print 'elapsed time = ', (end - start)
            #print 'Received data ', data
            if is_triggerring == 1:
                sent = self.sock.sendto(trigger_command, (slip_radio_ip, 9997));
                print 'Sent %s bytes back to %s' % (sent, slip_radio_ip)
                is_triggerring = 0
                trigger_command = ""
                
    def terminate(self):
        self._running = False
   

"========================================================================="
class tcpServerThread(threading.Thread):
    """On going development
    """
    def __init__(self, data_q):
        threading.Thread.__init__(self)
        self._running = True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((SERVER_IP, SERVER_PORT))
        self.sock.listen(1)
        self.data_q = data_q
    
    def terminate(self):
        self._running = False

    def run(self):
        print 'run TCP thread'
        self.conn, self.address = self.sock.accept()
        print 'address = ', self.address
        while self._running:
            data = self.conn.recv(1024) # buffer size is 1024 bytes
            #print "received message from ", addr, "with ", binascii.hexlify(data)
            if len(data) > 5:
                self.data_q.put(data)
            else:
                #print 'Missing data = ', data
                pass

"========================================================================="
class serialInputThread(threading.Thread):
    """Function: get serial input from sink node and transfer to collector-server
    via UDP connnection
    Input:
    Output:
    """
    def __init__(self, serial_name, baud_rate, database, table, data_q):
        super(serialInputThread, self).__init__()
        self.serial_name = serial_name 
        self.baud_rate = baud_rate
        self.data_q = data_q
        self.stopRequest = threading.Event()
        self.database = database
        self.table_tem = table + "Tem"
        self.table_acc = table + "Acc"
        self.table_ecg = table + "Ecg"

    def calculate_no_request_timeslot(self, new_sampling_rate):
        current_pkt_psecond = 2.0 # count based on time and seqno
        slot_frame_size = 17 
        current_ts_psecond = 1000/(slot_frame_size * 10) # 10ms each timeslot
        redundant_ts_psecond = current_ts_psecond - current_pkt_psecond
        # Extended parameters
        extended_pkt_psecond = new_sampling_rate * current_pkt_psecond
        extended_no_ts = math.ceil((extended_pkt_psecond + redundant_ts_psecond) / current_ts_psecond)
        return int(extended_no_ts - 1) # minus the current timeslot


    def write_command_trigger(self):
        new_sampling_rate = 4
        no_request_ts = 2
        duration = 120
        cmd = "tg " + str(new_sampling_rate) + " " + str(no_request_ts) + " " + str(duration) + "\r\n"
        #rev_cmd = str(trigger_command) + "\r\n"
        print "- Command send to sink: " + cmd
        #print "- command from udp server: ", rev_cmd
        count = 0
        while count < 20:
            self.serial_port.write(b''+cmd)
            #self.serial_port.write(b''+rev_cmd)
            time.sleep(0.01)
            count += 1
        print 'Done, count = ', count
        
    # Data format: systemtime, len, node_id, sensors..., sensortime
    def parserRawData(data):
        inputData = [val for val in data.split()]
        data_length = len(inputData)
        try:
            # Check length of message
            if data_length != int(inputData[1]) + 1: 
                # return anything with wrong received packet 
                print 'data_length = ', data_length, 'inputData = ', inputData
                return 0, inputData
            nodeId = int(inputData[5])
        except:
            nodeId = 0
            outData = np.int16([])
        return nodeId, inputData 
    
    def run(self):
        global is_triggerring
        print 'run Serial Input thread'
        stop = False
        # Create the Serial port
        try:
            self.serial_port = serial.Serial(self.serial_name,
                                             self.baud_rate)
        except:                                   
            print "Cannot open serial port"
            return
        # Start the Serial port
        print("- Serial: Listening to port %s at %s bps." % (self.serial_name, self.baud_rate))
        if not os.path.exists(DB_DIR):
            os.makedirs(DB_DIR)
        conn = sqlite3.connect(self.database)
        createDatabaseTem(conn, self.table_tem);
        createDatabaseAcc(conn, self.table_acc);
        createDatabaseEcg(conn, self.table_ecg);
        while not self.stopRequest.isSet():
            line = self.serial_port.readline()
            if line == "":
                print "empty line", line
            elif len(line) < 5:
                if is_triggerring:
                    inputData = [val for val in line.split()]
                    if inputData[0] == 'GTG':
                        print 'Got GTG command'
                        is_triggerring = 0
            else:
                #inputData = [val for val in line.split()]
                systemTime = int(round(time.time() * 1000))
                outputData = str(systemTime) + ' ' +  line
                if DEBUG:
                    print 'outPutData ' , outputData
                MAC_addr, outData = self.parserRawData(outputData)
                #start = stime.time()
                if MAC_addr == MAC_TEM:
                    updateSensorDataTem(conn, self.table_tem, outData)
                    self.data_q.put(outputData)
                elif MAC_addr == MAC_ACC:
                    updateSensorDataAcc(conn, self.table_acc, outData)
                    self.data_q.put(outputData)
                elif MAC_addr == MAC_ECG:
                    updateSensorDataEcg(conn, self.table_ecg, outData)
                    self.data_q.put(outputData)
                else:
                    print "Error" , outData
                #end = stime.time()
                #print 'elapsed time = ', (end - start)
                #print 'Received data ', data
            if is_triggerring:
                self.write_command_trigger()        

    def join(self, timeout=None):
        self.stopRequest.set()
        super(serialInputThread, self).join(timeout)


"========================================================================="
def displayWidgets():
    ## Create a grid layout to manage the widgets size and position
    layout = QtGui.QGridLayout()
    w.setLayout(layout)

    layout.addWidget(pACC,       0,1,2,1) # plot, row, col, wide, high
    layout.addWidget(pThrputAcc, 2,1,2,1)
    layout.addWidget(pPDRAcc,    4,1,2,1)

    layout.addWidget(pTEM,       0,2,2,1)
    layout.addWidget(pThrputTem, 2,2,2,1)
    layout.addWidget(pPDRTem,    4,2,2,1)

    layout.addWidget(pECG,       0,4,2,1)
    layout.addWidget(pThrputEcg, 2,4,2,1)
    layout.addWidget(pPDREcg,    4,4,2,1)
    #layout.addWidget(pJitter, 2,2,2,1)

"========================================================================="
"========================================================================="
def main():
    print 'DATABASE is "', DATABASE
    print 'TABLE is "', TABLE
    print 'Server: ', SERVER_IP, ':', SERVER_PORT
    if PREDICT:
        print 'Using trained model: ', TRAINED_MODEL
    else:
        print "No prediction"
    # Create a queue for transfer received data from udpServerThread to storingThread
    # and dataAnalysis tasks
    displayWidgets();
    data_q = Queue.Queue()
    BAUD_RATE = 115200
    if use_serial:
        print 'Run on serial port = ', SERIALPORT
        serialThread = serialInputThread(SERIALPORT, BAUD_RATE, DATABASE, TABLE, data_q)
        serialThread.daemon = True
        serialThread.start()
    elif useTCP:
        tcpThread = tcpServerThread(data_q)
        tcpThread.daemon = True
        tcpThread.start()
    else:
        udpThread = udpServerThread(DATABASE, TABLE, data_q)
        udpThread.daemon = True
        udpThread.start()
    # Display thread starts with input data_q from udpThread
    displayThread = displayDataThread(data_q)
    displayThread.daemon = True
    displayThread.start()



#Read in arguments
if __name__ == '__main__':
    parse_args(sys.argv[1:])


pygame.init()
overallDisplay = pygame.display.set_mode((Width+400,Height))
gameDisplay = overallDisplay.subsurface((0,0),(Width, Height))
networkPerformance = overallDisplay.subsurface((Width,0),(400,Height))
myFont = pygame.font.SysFont("monospace", 25)
stepFont = pygame.font.SysFont("Ubuntu",25,bold=True)
netperfFont = pygame.font.SysFont("Ubuntu", 14, bold=True)
ecg_pdr_label = netperfFont.render("ECG PDR(%) Thrput(bps) Jitter(ms)", 1, white)
acc_pdr_label = netperfFont.render("ACC PDR(%) Thrput(bps) Jitter(ms)", 1, white)
tem_pdr_label = netperfFont.render("TEM PDR(%) Thrput(bps) Jitter(ms)", 1, white)
startLabel = stepFont.render("Start Time",1,white)
activityLabel = stepFont.render("Activity",1,white)
pygame.display.set_caption('Activity Prediction')
clock = pygame.time.Clock()
timer = QtCore.QTimer()
timer.timeout.connect(updateSensor)
timer.start(0.03)

if __name__ == "__main__":
    main()
    if PREDICT:
        clf = joblib.load(TRAINED_MODEL)
        scaler = joblib.load(SCALER_STORINGMODEL)
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()
