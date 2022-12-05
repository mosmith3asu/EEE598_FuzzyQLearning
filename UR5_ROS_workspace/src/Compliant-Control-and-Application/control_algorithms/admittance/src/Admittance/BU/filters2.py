#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import Pose,Twist, Wrench
from geometry_msgs.msg import WrenchStamped 
from cartesian_state_msgs.msg import PoseTwist
from tf import TransformListener, listener
from std_msgs.msg import String
from tf import TransformListener
from tf.transformations import euler_from_quaternion, quaternion_from_euler,quaternion_multiply,quaternion_matrix,euler_matrix
from geometry_msgs.msg import Quaternion
import time 
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz
from scipy import signal,zeros#, random
from math import sin
# from scipy import zeros, signal, random

"""
def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def filter_sbs(data, b):
    z = zeros(b.size-1)
    result = zeros(data.size)
    for i, x in enumerate(data):
        result[i], z = signal.lfilter(b, 1, [x], zi=z)
    return result
    
def filter(data, b):
    result = signal.lfilter(b,1,data)
    return result


"""


def filter_sbs(data, b):
    z = np.zeros(b.size-1)
    result = np.zeros(data.size)
    for i, x in enumerate(data):
        result[i], z = signal.lfilter(b, 1, [x], zi=z)
    return result
    
def filter(data, b):
    result = signal.lfilter(b,1,data)
    return result

class RT_lowpass_butter(object):
	def __init__(self,fs,cutoff,order):
		self.cutoff = cutoff  
		self.fs = fs
		self.order = order
		#signal.butter(10, 15, 'hp', fs=1000, output='sos')
		
		self.b, self.a = butter(order, cutoff, fs=fs, btype='low', analog=False)
		#self.a = 1
		self.z = np.zeros(self.b.size-1)
		
	def sample(self,x):
		xfilt, self.z = signal.lfilter(self.b, self.a, [x], zi=self.z)
		#xfilt, self.z = signal.filtfilt(self.b, self.a, [x], zi=self.z)
		return xfilt
		

	



if __name__ == '__main__':
    #fs1 = 10
    #fs2 = 100
    #t = np.linspace(0,10,10000)
    #data = 1*np.sin(fs1*t) + 0.3*np.sin(fs2*t)
    #fs = 1000
    #t = np.linspace(0, 1, fs, False)  # 1 second
    #data = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*50*t)
    
    order = 5
    fs = 1000
    fc = 30  # Cut-off frequency of the filter
    w = fc / (fs / 2) # Normalize the frequency
    t = np.arange(1000) / fs
    

    signala = np.sin(2*np.pi*100*t) # with frequency of 100
    signalb = np.sin(2*np.pi*20*t) # frequency 20
    data = signala + signalb # Cumulative Signal
    
    #plt.plot(t, signala, label='siga')
    #plt.plot(t, signalb, label='sigb')
    
    
    # Filter ========================
    b, a = signal.butter(order, w, 'low')
    data_filt_static = signal.lfilter(b,a,data)
    #data_filt_static = signal.filtfilt(b, a, data)

    
    # Live Filter 1 ================
    filt = RT_lowpass_butter(fs,fc,order)
    data_filt_RT = [filt.sample(val) for val in data]
    #b = signal.firwin(order-1, 0.04)#fc,fs=fs)
    #data_filt_RT = filter_sbs(data, b)
    
    
    
    
    
    plt.plot(t, data, label='raw',lw=1)
    plt.plot(t, data_filt_static, label='f_static',lw=1)
    plt.plot(t, data_filt_RT, label='f_RT',lw=1)
    plt.legend()
    plt.show()
    
    
    
    
    
    
    # Live Filter 2 ================
    #order = 5
    #fc = 30
    #Wn = fc / (fs / 2) 
    #sos = signal.butter(order, [cutoff(s)], '[filter type]', output='sos')
    #sos = signal.butter(order, Wn , btype='low', output='sos')
    #sos = butter(order, fc, fs=fs, btype='low',output='sos')
    #f = IIR_filter(sos)
    #data_filt = [f.filter(sample) for sample in data]
    
    # Butterworth low-pass filter with frequency cutoff at 2.5 Hz
    #b, a = scipy.signal.iirfilter(4, Wn=2.5, fs=30, btype="low", ftype="butter")
    #b, a = scipy.signal.iirfilter(order, Wn=Wn, fs=fs, btype="low", ftype="butter")
    #yfilt = scipy.signal.lfilter(b, a, data) # apply filter once

    # PLOT =========
    plt.plot(t,data)
    plt.plot(t,data_filt)
    plt.show()
    
    #fc = 30  # Cut-off frequency of the filter
    #w = fc / (fs / 2) # Normalize the frequency
    #b, a = signal.butter(5, w, 'low')
    #output = signal.filtfilt(b, a, signalc)
    #plt.plot(t, output, label='filtered')
    #plt.legend()
    #plt.show()
    
		

#if __name__ == '__main__':
	#data = random.random(2000)
	#result = filter_sbs(data)	
	
	
	#plt.plot(data)
	#plt.plot(result)
	#plt.show()





"""
import scipy.signal
#from sklearn.metrics import mean_absolute_error as mae
from digitalfilter import LiveLFilter

# define lowpass filter with 2.5 Hz cutoff frequency
b, a = scipy.signal.iirfilter(4, Wn=2.5, fs=fs, btype="low", ftype="butter")
y_scipy_lfilter = scipy.signal.lfilter(b, a, yraw)

live_lfilter = LiveLFilter(b, a)
# simulate live filter - passing values one by one
y_live_lfilter = [live_lfilter(y) for y in yraw]

#print(f"lfilter error: {mae(y_scipy_lfilter, y_live_lfilter):.5g}")	


plt.figure(figsize=[6.4, 2.4])
plt.plot(ts, yraw, label="Noisy signal")
plt.plot(ts, y_scipy_lfilter, lw=2, label="SciPy lfilter")
plt.plot(ts, y_live_lfilter, lw=4, ls="dashed", label="LiveLFilter")

plt.legend(loc="lower center", bbox_to_anchor=[0.5, 1], ncol=3,
           fontsize="smaller")
plt.xlabel("Time / s")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()	
"""


