from __future__ import division
import datetime
import sys

__author__ = "Ngo Van Mao"
__copyright__ = "Copyright 2017, Singapore University of Technology and Design (SUTD)"
__credits__ = ["Ngo Van Mao"]
__license__ = "GNU GPLv3.0"
__version__ = "2.0.1"
__maintainer__ = "Ngo Van Mao"
__email__ = "vanmao_ngo@mymail.sutd.edu.sg"
__status__ = "Development"

from harCommon import *

SENDING_RATE = 500 # ms
MAX_SZ = 60
class NetPerf:
    def __init__(self, node_id):
        self.node_id = node_id
        self.rec_pkt = 0
        self.lost_pkt = 0
        self.systemtime = []
        self.state = []
        self.seqno = []
        self.rssi_level = []
        self.mac_uni_ok = 0 
        self.mac_uni_noack = 0 
        # jitter 
        self.duplicates = 0;
        self.throughput = 0
        self.jitter = [0]
        self.mean_jitter = 0
        self.upper_jitter = 0
        self.lower_jitter = 0
        self.min_seqno = sys.maxint
        self.max_seqno = -sys.maxint - 1
        self.seqno_delta = 0
        self.mac_uni_ok_delta = 0
        self.mac_uni_noack_delta = 0
        self.max_mac_uni_ok = -sys.maxint - 1
        self.max_mac_uni_noack = -sys.maxint - 1
        self.node_restart_count = 0
        self.pdr = 100.00
        self.prr = 100.00
        self.elapsed_time = 1
        self.start_time = float('inf') #sys.maxsize
        self.end_time = 0
        self.data_len = 0
        self.curr_state = STATE_NORMAL
        self.last_systemtime = 0
        # sliding window pdr and throughput
        self.time_wind = [0]*MAX_SZ
        self.pkt_wind = [0]*MAX_SZ
        self.lpk_wind = [0]*MAX_SZ
        self.dup_wind = [0]*MAX_SZ

    def resetParam(self):
        self.rec_pkt = 0
        self.lost_pkt = 0
        self.throughput = 0
        self.jitter = [0]
        self.mean_jitter = 0
        self.upper_jitter = 0
        self.lower_jitter = 0
        self.pdr = 100.00
        self.prr = 100.00
        self.elapsed_time =1 
        self.start_time = float('inf') #sys.maxsize
        self.end_time = 0
        self.duplicates = 0;
        # sliding window pdr and throughput
        self.time_wind = [0]*MAX_SZ
        self.pkt_wind = [0]*MAX_SZ
        self.lpk_wind = [0]*MAX_SZ
        self.dup_wind = [0]*MAX_SZ

    def getNodeId(self):
        return self.node_id

    def updateNetPerf(self, data):
        """sensor data:
        index  0         , 1  , 2    , 3    , 4     , 5      , 6    , 7  , 8
        data =[systemtime, len, time1, time2, state, node_id, seqno, hop, rssi,
               ax1, ay1, az1, ax2, ay2, az2, ax3, ay3, az3, ax4, ay4, az4, \
               ax5, ay5, az5, ax6, ay6, az6, ax7, ay7, az7, ax8, ay8, az8, \
               mac_stat] \
        """
        systime = int(data[0])
        sch_state = int(data[4])  & 0xff
        scale_sending_rate = (int(data[4]) >> 8) & 0xff
        self.nod_id = int(data[5])
        sqn     = int(data[6]) 
        hop     = int(data[7])
        rssi    = int(data[8])
        mac_pkt = int(data[-1]) #len(data)-1]) 
        mac_lost = mac_pkt & 0xff
        mac_sent = (mac_pkt >> 8) & 0xff

        s = sqn + self.seqno_delta
        tem_mac_uni_ok = mac_sent + self.mac_uni_ok_delta
        tem_mac_uni_noack = mac_lost + self.mac_uni_noack_delta
        is_fresh = True
        if len(data[8:]) > self.data_len:
            self.data_len = len(data[8:]) # from 8 because of including seqno
            #print 'self.data_len = ', self.data_len

        # Check duplicate
        n = len(self.seqno)
        if s <= self.max_seqno:
            if n > 5:
                start_sq = n - 5
            else:
                start_sq = 0
            for i in range(start_sq, n):
                if(self.seqno[i] != s):
                    pass
                elif abs(self.systemtime[i] - systime) > 180000: #3s
                    print 'Too OLD packet >3s'
                    pass
                else:
                    print 'duplicated'
                    is_fresh = False
                    # Verify that the packet data is a real duplicate
                    # TODO: will implement compare full message
                    self.duplicates +=1
                    self.dup_wind.append(1); del self.dup_wind[0]

        if is_fresh:
            if sch_state != self.curr_state:
                self.curr_state = sch_state 
            #    #Reset parameter when changing to a new state
            #    self.resetParam() 

            #print 'self.seqno.size() = ', n
            if n == 0:
                self.last_systemtime = systime 
            if n > 0:
                delta_receiver_time = systime - self.last_systemtime #jitter
                scale_sending_rate = 1
                delta_sender_time = SENDING_RATE / scale_sending_rate
                time_diff = delta_receiver_time - delta_sender_time
                self.last_systemtime = systime
                self.jitter.append(time_diff)
                #self.elapsed_time = systime - self.systemtime[0]
                if systime > self.end_time:
                    self.end_time = systime
                if systime < self.start_time:
                    self.start_time = systime
                    print 'start_time = ', self.start_time, " = ",\
                      datetime.datetime.fromtimestamp(self.start_time/1000).strftime("%Y-%m-%d %H:%M:%S") 
                self.elapsed_time = self.end_time - self.start_time
                if self.max_seqno - s > 2:
                    #Handle sequence number overflow
                    self.seqno_delta = self.max_seqno + 1
                    s = self.seqno_delta + sqn
                    if sqn > 127:
                        # Sequence number restarted at 128 (to separate node restarts
                        # from sequence number overflow
                        sqn -= 128
                        self.seqno_delta -= 128
                        s -= 128
                    else:
                        self.node_restart_count +=1
                        self.resetParam()
                    if sqn > 0:
                        self.lost_pkt += sqn
                        self.lpk_wind.append(sqn); del self.lpk_wind[0]
                    else:
                        self.lpk_wind.append(0); del self.lpk_wind[0]
                elif s > (self.max_seqno + 1):
                    self.lost_pkt += s - (self.max_seqno + 1)
                    # DEBUG can be useful
                    #print 'lost = ', self.lost_pkt , 'node_id = ', self.node_id
                    self.lpk_wind.append(s - (self.max_seqno + 1)); del self.lpk_wind[0]
                else:
                    self.lpk_wind.append(0); del self.lpk_wind[0]

                if self.max_mac_uni_ok - tem_mac_uni_ok > 2:
                    #Handle sequence number overflow
                    self.mac_uni_ok_delta = self.max_mac_uni_ok + 1
                    tem_mac_uni_ok = self.mac_uni_ok_delta + mac_sent 
                if self.max_mac_uni_noack - tem_mac_uni_noack > 2:
                    #Handle sequence number overflow
                    self.mac_uni_noack_delta = self.max_mac_uni_noack + 1
                    tem_mac_uni_noack = self.mac_uni_noack_delta + mac_lost 

            self.rec_pkt += 1
            self.time_wind.append(systime); del self.time_wind[0]
            self.dup_wind.append(0); del self.dup_wind[0]
            self.pkt_wind.append(1); del self.pkt_wind[0]

            if self.elapsed_time  > 3000:
                #self.throughput = ((self.rec_pkt + self.duplicates) * 16 * self.data_len)\
                #                    /(self.elapsed_time/1000)
                self.mean_jitter = np.mean(self.jitter)
                self.upper_jitter = self.mean_jitter + np.std(self.jitter)
                self.lower_jitter = self.mean_jitter - np.std(self.jitter)
               # self.throughput = ((sum(self.pkt_wind) + sum(self.dup_wind)) * 16 * self.data_len)\
               #                   /abs((self.time_wind[MAX_SZ-1] - self.time_wind[0])/1000.00)
               # self.pdr = 100.00* sum(self.pkt_wind)  / (sum(self.pkt_wind) + sum(self.lpk_wind))
                
            if s < self.min_seqno:
                self.min_seqno = s
            if s > self.max_seqno:
                self.max_seqno = s
            if tem_mac_uni_ok > self.max_mac_uni_ok:
                self.max_mac_uni_ok = tem_mac_uni_ok
            if tem_mac_uni_noack > self.max_mac_uni_noack:
                self.max_mac_uni_noack = tem_mac_uni_noack
        self.throughput = ((sum(self.pkt_wind) + sum(self.dup_wind)) * 16 * self.data_len)\
                          /abs((self.time_wind[MAX_SZ-1] - self.time_wind[0])/1000.00)
        self.pdr = 100.00* sum(self.pkt_wind)  / (sum(self.pkt_wind) + sum(self.lpk_wind))
        # Check restart node
        #self.pdr = 100 * self.rec_pkt / (self.rec_pkt + self.lost_pkt)
        self.systemtime.append(systime)
        self.state.append(sch_state)
        self.seqno.append(s)
        self.rssi_level.append(rssi)
        self.mac_uni_ok = tem_mac_uni_ok
        self.mac_uni_noack = tem_mac_uni_noack

    def getPrr(self):
        if self.mac_uni_ok > 0:
            self.prr = 100 * (self.mac_uni_ok)/(self.mac_uni_noack + self.mac_uni_ok)
        else:
            self.prr = 0
        return self.prr


