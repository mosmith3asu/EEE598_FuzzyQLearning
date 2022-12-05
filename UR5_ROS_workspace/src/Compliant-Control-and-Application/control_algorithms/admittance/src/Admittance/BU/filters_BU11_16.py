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
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import math
plt.rcParams["figure.figsize"] = 10,5
plt.rcParams["font.size"] = 16

class RT_lowpass_butter(object):
	def __init__(self,fs,cutoff,order,enable=True):
		self.enable = enable
		self.cutoff = cutoff  
		self.fs = fs
		self.order = order
		self.b, self.a = butter(order, cutoff, fs=fs, btype='low', analog=False)
		#self.a = 1
		self.z = np.zeros(self.b.size-1)
		
	def sample(self,x):
		if not self.enable: return x
		xfilt, self.z = signal.lfilter(self.b, self.a, [x], zi=self.z)
		return xfilt
	
	def test(self,yraw):
		#if not self.enable: return yraw
		yfilt = [self.sample(val) for val in yraw]
		return yfilt
		
class LowPass_Filter(object):
	def __init__(self,fs):
		self.fs = fs
		self.dt = 1/fs
		w0 = 2*np.pi*5; # pole frequency (rad/s)
		num = w0        # transfer function numerator coefficients
		den = [1,w0]    # transfer function denominator coefficients
		self.lowPass = signal.TransferFunction(num,den) # Transfer function

		self.b,self.a = self.get_filt_coeff()
		self.last_yraw = None
		self.last_yfilt = None
		self.memory = {'raw': [], 'filt': [],'enable':True}
		
	def get_filt_coeff(self,verbose=False):
		# Unpack
		fs = self.fs
		dt = self.dt
		lowPass = self.lowPass
		
		# Discrete TF
		discreteLowPass = lowPass.to_discrete(dt,method='gbt',alpha=0.5) 
		
		# Filter Coefficients
		# The coefficients from the discrete form of the filter transfer function (but with a negative sign)
		b = discreteLowPass.num;
		a = -discreteLowPass.den;
		if verbose:
			print(discreteLowPass)
			print("Filter coefficients b_i: " + str(b))
			print("Filter coefficients a_i: " + str(a[1:]))
		return b,a
	
	def test_signal(self,t,Yraw,plot_result=False):
		Yfilt = np.zeros(len(Yraw));
		for i in range(len(Yraw)): #for i in range(3,len(Yraw)):
			Yfilt[i] = self.sample(Yraw[i])	
	
		
		if plot_result: 
			# Plot the signal
			plt.figure()
			plt.plot(t,Yraw);
			plt.plot(t,Yfilt);
			plt.ylabel("$y(t)$")
			plt.xlim([min(t),max(t)]);

			# Generate Fourier transform
			Yhat = np.fft.fft(Yraw);
			Yfilthat = np.fft.fft(Yfilt)
			fcycles = np.fft.fftfreq(len(t),d=1.0/self.fs)
			
			# Plot spectrum
			plt.figure()
			plt.plot(fcycles,np.absolute(Yhat));
			plt.plot(fcycles,np.absolute(Yfilthat));
			plt.xlim([-100,100]);
			plt.xlabel("$\omega$ (cycles/s)");
			plt.ylabel("$|\hat{y}|$");
			plt.title('Filter Result')
			#plt.show()
		return Yfilt
		
	def sample(self,yraw):
		if self.last_yraw is None or self.last_yfilt is None: 
			self.last_yraw = yraw
			self.last_yfilt = 0
			if self.memory['enable']:
				self.memory['raw'].append(self.last_yraw )
				self.memory['filt'].append(self.last_yfilt)
			return 0
		else:			
			yfilt = self.a[1]*self.last_yfilt + self.b[0]*yraw + self.b[1]*self.last_yraw
			self.last_yraw = yraw
			self.last_yfilt = yfilt
			if self.memory['enable']:
				self.memory['raw'].append(self.last_yraw )
				self.memory['filt'].append(self.last_yfilt)
			return yfilt
			
			
	def compare_raw2filt(self,t=None):
		if self.memory['enable']:
			
			Yraw = self.memory['raw']
			Yfilt =self.memory['filt']
			if t is None: t=np.arange(len(Yraw))
			
			# Plot the signal
			plt.figure()
			plt.plot(t,Yraw);
			plt.plot(t,Yfilt);
			plt.ylabel("$y(t)$")
			plt.xlim([min(t),max(t)]);

			# Generate Fourier transform
			Yhat = np.fft.fft(Yraw);
			Yfilthat = np.fft.fft(Yfilt)
			fcycles = np.fft.fftfreq(len(t),d=1.0/self.fs )

			plt.figure()
			plt.plot(fcycles,np.absolute(Yhat));
			plt.plot(fcycles,np.absolute(Yfilthat));
			plt.xlim([-100,100]);
			plt.xlabel("$\omega$ (cycles/s)");
			plt.ylabel("$|\hat{y}|$");
			plt.title('Filter Result')
			#plt.show()
			
	##########################################
	# ANALYSIS ################################
	def signal_preview(self,t,y):
		# Plot the signal
		plt.figure()
		plt.plot(t,y);
		plt.ylabel("$y(t)$");
		plt.xlabel("$t$ (s)");
		plt.xlim([min(t),max(t)]);
		plt.title('SIGNAL: Raw Data')
		
	def signal_spectrum(self,t,y):
		# Compute the Fourier transform
		yhat = np.fft.fft(y);
		fcycles = np.fft.fftfreq(len(t),d=1.0/self.fs); # the frequencies in cycles/s
		# Plot the power spectrum
		plt.figure()
		plt.plot(fcycles,np.absolute(yhat));
		plt.xlim([-100,100]);
		plt.xlabel("$\omega$ (cycles/s)");
		plt.ylabel("$|\hat{y}|$");
		plt.title('SIGNAL: Power Spectrum')

	def filter_magplt(self):
		# Unpack
		fs = self.fs
		dt = self.dt
		lowPass = self.lowPass
	
		# Generate the bode plot
		signalFreq = [2,50]
		w = np.logspace( np.log10(min(signalFreq)*2*np.pi/10), np.log10(max(signalFreq)*2*np.pi*10), 500 )
		w, mag, phase = signal.bode(lowPass,w)
		
		# Magnitude plot
		plt.figure()
		plt.semilogx(w, mag)
		for sf in signalFreq:
			plt.semilogx([sf*2*np.pi,sf*2*np.pi],[min(mag),max(mag)],'k:')
		plt.ylabel("Magnitude ($dB$)")
		plt.xlim([min(w),max(w)])
		plt.ylim([min(mag),max(mag)])
		plt.title('FILTER: Magnitude Plot')
	def filter_phaseplt(self):
		# Unpack
		fs = self.fs
		dt = self.dt
		lowPass = self.lowPass
		
		# Generate the bode plot
		signalFreq = [2,50]
		w = np.logspace( np.log10(min(signalFreq)*2*np.pi/10), np.log10(max(signalFreq)*2*np.pi*10), 500 )
		w, mag, phase = signal.bode(lowPass,w)
		
		# Phase plot
		plt.figure()
		plt.semilogx(w, phase)  # Bode phase plot
		plt.ylabel("Phase ($^\circ$)")
		plt.xlabel("$\omega$ (rad/s)")
		plt.xlim([min(w),max(w)])
		plt.title('FILTER: Phase Plot')

	

def gen_fake_signal(fs):
	#"""
	f1,f2 =  100,20
	t = np.arange(1000) / fs
	signala = np.sin(2*np.pi*f1*t) # with frequency of 100
	signalb = np.sin(2*np.pi*f2*t) # frequency 20
	y = signala + signalb # Cumulative Signal
	
	"""
	# Generate a signal
	# samplingFreq = 1000; # sampled at 1 kHz = 1000 samples / second
	tlims = [0,1]        # in seconds
	signalFreq = [2,50]; # Cycles / second
	signalMag = [1,0.2]; # magnitude of each sine
	t = np.linspace(tlims[0],tlims[1],(tlims[1]-tlims[0])*fs)
	y = signalMag[0]*np.sin(2*math.pi*signalFreq[0]*t) + signalMag[1]*np.sin(2*math.pi*signalFreq[1]*t)
	"""
	return t,y
	

def test_main():
	# Generate a signal
	samplingFreq = 1000; # sampled at 1 kHz = 1000 samples / second
	tlims = [0,1]        # in seconds
	signalFreq = [2,50]; # Cycles / second
	signalMag = [1,0.2]; # magnitude of each sine
	t = np.linspace(tlims[0],tlims[1],(tlims[1]-tlims[0])*samplingFreq)
	y = signalMag[0]*np.sin(2*math.pi*signalFreq[0]*t) + signalMag[1]*np.sin(2*math.pi*signalFreq[1]*t)
	
	filt = LowPass_Filter(samplingFreq)
	#filt.signal_preview(t,y)
	#filt.signal_spectrum(t,y)
	#filt.filter_phaseplt()
	#filt.filter_magplt()
	
	yfilt = filt.test_signal(t,y)
	filt.compare_raw2filt()
	plt.show()
def test_main2():
	# Packages and adjustments to the figures

	#plt.rcParams.update({"text.usetex": True,"font.family": "sans-serif","font.sans-serif": ["Helvetica"]})
	
	
	
	
	#fs = 1000
    #fc = 30  # Cut-off frequency of the filter
    #w = fc / (fs / 2) # Normalize the frequency
    #t, data = gen_fake_signal(fs)

	# Compute the Fourier transform
	yhat = np.fft.fft(y);
	fcycles = np.fft.fftfreq(len(t),d=1.0/samplingFreq); # the frequencies in cycles/s

	# Plot the signal
	plt.figure()
	plt.plot(t,y);
	plt.ylabel("$y(t)$");
	plt.xlabel("$t$ (s)");
	plt.xlim([min(t),max(t)]);

	# Plot the power spectrum
	plt.figure()
	plt.plot(fcycles,np.absolute(yhat));
	plt.xlim([-100,100]);
	plt.xlabel("$\omega$ (cycles/s)");
	plt.ylabel("$|\hat{y}|$");
	
	
	
	# Low-pass filter
	w0 = 2*np.pi*5; # pole frequency (rad/s)
	num = w0        # transfer function numerator coefficients
	den = [1,w0]    # transfer function denominator coefficients
	lowPass = signal.TransferFunction(num,den) # Transfer function

	# Generate the bode plot
	w = np.logspace( np.log10(min(signalFreq)*2*np.pi/10), np.log10(max(signalFreq)*2*np.pi*10), 500 )
	w, mag, phase = signal.bode(lowPass,w)

	# Magnitude plot
	plt.figure()
	plt.semilogx(w, mag)
	for sf in signalFreq:
		plt.semilogx([sf*2*np.pi,sf*2*np.pi],[min(mag),max(mag)],'k:')
	plt.ylabel("Magnitude ($dB$)")
	plt.xlim([min(w),max(w)])
	plt.ylim([min(mag),max(mag)])

	# Phase plot
	plt.figure()
	plt.semilogx(w, phase)  # Bode phase plot
	plt.ylabel("Phase ($^\circ$)")
	plt.xlabel("$\omega$ (rad/s)")
	plt.xlim([min(w),max(w)])
	plt.show()
	
	
	
	# Discrete TF
	dt = 1.0/samplingFreq;
	discreteLowPass = lowPass.to_discrete(dt,method='gbt',alpha=0.5)
	print(discreteLowPass)
	
	# Filter Coefficients
	# The coefficients from the discrete form of the filter transfer function (but with a negative sign)
	b = discreteLowPass.num;
	a = -discreteLowPass.den;
	print("Filter coefficients b_i: " + str(b))
	print("Filter coefficients a_i: " + str(a[1:]))

	# Filter the signal
	yfilt = np.zeros(len(y));
	for i in range(3,len(y)):
		yfilt[i] = a[1]*yfilt[i-1] + b[0]*y[i] + b[1]*y[i-1];
		
	# Plot the signal
	plt.figure()
	plt.plot(t,y);
	plt.plot(t,yfilt);
	plt.ylabel("$y(t)$")
	plt.xlim([min(t),max(t)]);

	# Generate Fourier transform
	yfilthat = np.fft.fft(yfilt)
	fcycles = np.fft.fftfreq(len(t),d=1.0/samplingFreq)

	plt.figure()
	plt.plot(fcycles,np.absolute(yhat));
	plt.plot(fcycles,np.absolute(yfilthat));
	plt.xlim([-100,100]);
	plt.xlabel("$\omega$ (cycles/s)");
	plt.ylabel("$|\hat{y}|$");
	plt.show()


	
def butter_main():
    order = 2
    
    fs = 1000
    fc = 30  # Cut-off frequency of the filter
    w = fc / (fs / 2) # Normalize the frequency
    t, data = gen_fake_signal(fs)

    
    
    # Static Filter ========================
    #b, a = signal.butter(order, w, 'low')
    #data_filt_static = signal.lfilter(b,a,data)
    

    
    # Live Filter 1 ================
    filt = RT_lowpass_butter(fs,fc,order)
    data_filt_RT = filt.test(data) 
    #data_filt_RT = [filt.sample(val) for val in data]

    # Plot Result ===================
    plt.plot(t, data, label='raw',lw=1)
    #plt.plot(t, data_filt_static, label='f_static',lw=1)
    plt.plot(t, data_filt_RT, label='f_RT',lw=1)
    plt.plot(t, signalb, label='goal',lw=1)
    plt.legend()
    plt.show()
if __name__ == '__main__':    

    
    test_main()
    

		

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


