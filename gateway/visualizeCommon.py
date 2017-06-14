#!/usr/bin/python
"""Contains some general classes used to display data collected from sensors via pyqtgraph. 
"""

import pyqtgraph as pg
import datetime
from pyqtgraph.ptime import time
from pyqtgraph import ViewBox
from harCommon import *
from pyqtgraph import AxisItem
from pyqtgraph.Qt import QtGui, QtCore

__author__ = "Sanil Kumar Ashwin, Ngo Van Mao"
__copyright__ = "Copyright 2016, Singapore University of Technology and Design (SUTD)"
__credits__ = ["Sanil Kumar Ashwin", "Ngo Van Mao"]
__license__ = "GNU GPLv3.0"
__version__ = "1.0.1"
__maintainer__ = "Ngo Van Mao"
__email__ = "vanmao_ngo@mymail.sutd.edu.sg, elvenashwin@gmail.com"
__status__ = "Production"

class TimeAxisItem(AxisItem):
    """Extends AxisItem, reads unix timestamp values and displays 
    them as formatted date time values

    Usage:    Input instance as argument in PlotWidget construction
    Ex: pg.PlotWidget(axisItems={'bottom': TimeAxisItem(orientation='bottom')})
    """
    
    def __init__(self, *args, **kwargs):
        super(TimeAxisItem,self).__init__(*args, **kwargs)

    def tickStrings(self, values, scale, spacing):
        return [datetime.datetime.fromtimestamp(value/1000).strftime('%H:%M:%S') for value in values]

class SensorPlot:
    """A SensorPlot encapsulates all the related graphs pertaining to a specific sensor
    IE the LH sensor, or the RF sensor.
    """
    
    def __init__(self,name, id,magnetometer = True, length=1000):
        """Name refers to the string identifier for this sensor plot
        ID is the numerical identifier for this sensor plot (found in harCommon)
        magnetometer is a flag indicating whether the magnetometer graph should be created
        Length is the length of the x-axis (ie the time stamp array) in the dataseries
        """
        self.name = name # ie 'LF', 'RH' etc
        self.id = id
        self.accelerometer = SensorComponent('Accelerometer', name, FULL_ACC, length)
        self.gyroscope = SensorComponent('Gyroscope', name, FULL_GYRO, length)
        if magnetometer:
            self.magnetometer = SensorComponent('Magnetometer', name, FULL_MAG,length)
            self.mag = self.magnetometer
        self.acc = self.accelerometer
        self.gyr = self.gyroscope
        self.components = [self.acc, self.gyr]
        if magnetometer:
            self.components.append(self.mag)
        self.ts = [0]*length
        self.is_mag = magnetometer
        self.length = length
        self.prediction = []
        self.packetCount = 0
        indexOfID = np.where(ID==id)[0][0]      #get index of number
                                                #because CID is index 0, so shifted to the right
        self.flag = 2**(indexOfID-1)
        
    def update(self):
        """Updates all the curves of all the graphs of this SensorPlot"""
        for component in self.components:
            component.update(self.ts)
    
    def normalize_timestamp(self, initval):
        """Normalizes the timestamp
        
        Initially, the timestamp array of all the graphs is an array of 0s
        Hence, the x-axis on all the graphics will display 7:30 am, 
        1 January 1970 (The Epoch)
        After the first value is read, the previous values must be set to 
        small intervals before it, so the x-axis will display correctly.
        
        Input: initval, the first value read in
        """
        count = self.length
        c_ts = initval
        while count > 0:
            count -= 1
            c_ts-=1000/32
            self.ts[count] = c_ts
    
    def append_data(self, timestamp, destructive = True,**data):
        """Adds the data for whatever component to the various graphs, at the specific
        timestamp (in milliseconds).
        If destructive is set to True, first value of every curve will be deleted
        as new value is inserted (resulting in the curve "moving" used in realtime
        display)
        The data for each component graph needs have 3 elements, for the x,y and z
        axes respectively
        The component graphs are mag/magnetometer, acc/accelerometer, and
        gyr/gyroscope
        example: append_data(1479969986000, mag= [110,-10,4], acc=[5,6,7])
        """

        self.ts.append(timestamp)
        for name in data:
            value = data[name]
            if name in ('acc','accelerometer'):
                self.acc.append_data(value, destructive)
            elif name in ('gyr','gyroscope'):
                self.gyr.append_data(value, destructive)
            elif name in ('mag','magnetometer'):
                self.mag.append_data(value, destructive)
        if destructive:
            del self.ts[0]
class SensorComponent:#IE ARH, ALH
    """Refers to an individual graph: ie the representation of the 
    accelerometer, gyroscope, magnetometer or other component of a 
    sensor.
    """
    
    def __init__(self, name, sensor,range,length=1000):
        """Name refers to the name of this component
        sensor refers to the name of the sensor (IE LH, RF)
        range will be used for the minimum and maximum on this graph, as 
        [range,-range]
        length is the length of the x-axis timestamp array, set to 1000 by 
        default
        """ 
        self.plot_widget = pg.PlotWidget(
                     axisItems={'bottom': TimeAxisItem(orientation='bottom')})
        self.pw = self.plot_widget
        self.pw.addLegend()
        self.x = ComponentAxis('x', self.pw, 'r', length)
        self.y = ComponentAxis('y', self.pw, 'g', length)
        self.z = ComponentAxis('z', self.pw, 'b', length)
        self.m = ComponentAxis('m', self.pw, 'y', length) # magnitude
        self.axes = [self.x, self.y, self.z, self.m]

        self.pw.setTitle(name + ' ' + sensor)
        self.pw.setYRange(-range, range, padding=0)
        self.pw.enableAutoRange(ViewBox.XAxis)
        self.pw.enableAutoRange(ViewBox.YAxis)
    
    def update(self,xaxis):
        """Updates all the curves in this graph, with a given timestamp array i
        and their stored data series.
        """        
        for axis in self.axes:
            axis.update(xaxis)
    
    def append_data(self, data, destructive=True):
        """Appends the data to each of the curves
        First element appended to the x-axis series, second to the y-axis series
        and third to the z-axis series
        If destructive is set to True, first element of all series will be 
        deleted which will result in moving curve
        """
        self.x.add_data(data[0], destructive)
        self.y.add_data(data[1], destructive)
        self.z.add_data(data[2], destructive)
        self.m.add_data((data[0]**2+data[1]**2+data[2]**2)**0.5, destructive)

class ComponentAxis:
    """An axis of a component of a sensor.
    EG the y-axis of the magnetometer of the LH sensor
    """
    
    def __init__(self, name, plotwidget, color,length=1000):
        """Will draw a curve with the given plotwidget and a pen specified
        by the given color.
        Length is the length of the timestamp axis (the x-xis)
        """
        self.curve = plotwidget.plot(pen=color,name=name)
        self.data = [0] * length
        self.name = name
    
    def add_data(self, dataitem, destructive=True):
        """Appends data series with new dataitem
        Deletes the first data item if destructive is set to True
        , a setting used in moving curve (realtime display)
        """
        self.data.append(dataitem)
        if destructive:
            del self.data[0]
    
    def update(self, xaxis):
        """Updates curve with the self.data as the y-value and the 'xaxis' argument as
        the x-value
        """
        self.curve.setData(xaxis,self.data)


def displayWidgets(full, sensor_plot, qwidget):
    """Displays the graphs passed to it on the given QWidget in a standard
       format (LH/RH/LF/RF, left to right)
    
    --Input--
    full        a boolean indicating whether all 4 sensors (or just 2) need 
                to be displayed
    sensor_plot a dictionary of SensorPlot instances (with RF and LH
                always present as keys, and RH and LF as keys if full=True)
    qwidget     A QWidget instance
    
    --Effects--
    Will display all the graphs of all the sensor_plots (will check if
    magnetometer needs to be displayed, by itself)
    
    Output:   None
    """
    w = qwidget 
    ## Create a grid layout to manage the widgets size and position
    layout = QtGui.QGridLayout()
    w.setLayout(layout)
    RF = sensor_plot['RF']
    LH = sensor_plot['LH']
    layout.addWidget(LH.acc.pw, 0,1,2,1)
    layout.addWidget(LH.gyr.pw, 2,1,2,1)
    if LH.is_mag:
        layout.addWidget(LH.mag.pw, 4,1,2,1)
    
    layout.addWidget(RF.acc.pw, 0,4,2,1)
    layout.addWidget(RF.gyr.pw, 2,4,2,1)
    if RF.is_mag:
        layout.addWidget(RF.mag.pw, 4,4,2,1)

    if full:
        RH = sensor_plot['RH']
        LF = sensor_plot['LF']
        layout.addWidget(RH.acc.pw, 0,2,2,1)
        layout.addWidget(RH.gyr.pw, 2,2,2,1)
        if RH.is_mag: 
            layout.addWidget(RH.mag.pw, 4,2,2,1)
        
        layout.addWidget(LF.acc.pw, 0,3,2,1)
        layout.addWidget(LF.gyr.pw, 2,3,2,1)
        
        if LH.is_mag:
            layout.addWidget(LF.mag.pw, 4,3,2,1)
