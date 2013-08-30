'''
Implement continuous wavelet transform for python
'''

import scipy.signal
import numpy as np
import pywt
import matplotlib.pyplot as plt

class CWT:
	def __init__(self , waveletType):
		self.wavelet = pywt.Wavelet(waveletType)
    	[self.phi, self.psi, self.x] = wavelet.wavefun(4)
    	# Compute the central frequency
        fft = np.fft.fft(psi)
    	freqs = np.fft.fftfreq(len(fft) , x[1]-x[0])
        idx = np.nonzero(freqs >= 0)[0] # Only consider positive frequencies
        freqs = freqs[idx]
        fft = fft[idx]
        idx = np.nonzero(fft == np.max(fft))[0]
        self.centralFreq = freqs[idx][0]

    def centralFrequency():
        return self.centralFreq

    def getWavelet(self, scale, dt, time, data):
        basis = x * self.scale
        basis = basis * dt
        yVals = self.psi/np.sqrt(scale)
        delta = basis[1] - basis[0]
        if max(basis) < max(time):
            basis = np.arange(0,len(yVals))*delta
            signal = data
            modTime = time
            idx = np.nonzero(modTime <= np.max(basis))[0]
            return np.interp(modTime[idx] , basis, yVals), signal, modTime
        else:
            num_pads = np.ceil((max(basis) - max(time))/dt)
            num_pads = max(0,num_pads)
            postPads = np.zeros(num_pads)
            signal = np.concatenate((data,postPads))
            modTime = np.arange(0,len(signal))*dt
            return np.interp(modTime , basis, yVals), signal

    def transform(self, data, dt, scales, scaleFactor):
        time = np.array(range(len(data)))*dt
        output = np.zeros([len(scales), len(data)])
        for idx, scale in enumerate(scales):
            wavelet, signal = getWavelet(scale , dt, time, data)
            output[idx, :] = scipy.signal.convolve(signal, wavelet, mode='same')[:len(data)]
        return output

    def scaleToFreq(self , scale, dt = 1):
        return self.centralFreq/(scale * dt)

    def freqToScale(self, freq, dt = 1):
        return self.centralFreq/(dt * freq)

    def plot(self, data, dt, scales, coefs):
        time = np.arange(0,len(data))*dt
        X, Y = np.meshgrid(time, scales)
        plt.subplot(2,1,1)
        plt.plot(time , data)
        plt.subplot(2,1,2)
        plt.pcolor(X, Y, coefs**2)
        plt.show()

    def savePlot(self, data, dt, scales, coefs, fname):
        time = np.arange(0,len(data))*dt
        X, Y = np.meshgrid(time, scales)
        plt.subplot(2,1,1)
        plt.plot(time , data)
        plt.subplot(2,1,2)
        plt.pcolor(X, Y, coefs**2)
        plt.savefig(fname)
        plt.close()
