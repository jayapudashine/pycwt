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
        [self.phi, self.psi, self.x] = self.wavelet.wavefun(4)
        # Compute the central frequency
        fft = np.fft.fft(self.psi)
        freqs = np.fft.fftfreq(len(fft) , self.x[1] - self.x[0])
        idx = np.nonzero(freqs >= 0)[0] # Only consider positive frequencies
        freqs = freqs[idx]
        fft = fft[idx]
        idx = np.nonzero(fft == np.max(fft))[0]
        self.centralFreq = freqs[idx][0]

    def centralFrequency():
        return self.centralFreq

    def getWavelet(self, scale, dt, time, data):
        basis = self.x * scale
        basis = basis * dt
        yVals = self.psi/np.sqrt(scale)
        delta = basis[1] - basis[0]
        if max(basis) < max(time):
            basis = np.arange(0,len(yVals))*delta
            signal = data
            modTime = time
            idx = np.nonzero(modTime <= np.max(basis))[0]
            return np.interp(modTime[idx] , basis, yVals), signal
        else:
            num_pads = np.ceil((max(basis) - max(time))/dt)
            num_pads = max(0,num_pads)
            postPads = np.zeros(num_pads)
            signal = np.concatenate((data,postPads))
            modTime = np.arange(0,len(signal))*dt
            return np.interp(modTime , basis, yVals), signal

    def transform(self, data, dt, scales):
        time = np.array(range(len(data)))*dt
        output = np.zeros([len(scales), len(data)])
        print 'Working on',
        for idx, scale in enumerate(scales):
            print idx,
            wavelet, signal = self.getWavelet(scale , dt, time, data)
            output[idx, :] = scipy.signal.convolve(signal, wavelet, mode='same')[:len(data)]
        print 'Done'
        return output

    def scaleToFreq(self , scale, dt = 1):
        return self.centralFreq/(scale * dt)

    def freqToScale(self, freq, dt = 1):
        return self.centralFreq/(dt * freq)

    def plot(self, data, dt, periods, coefs):
        time = np.arange(0,len(data))*dt
        X, Y = np.meshgrid(time, periods)
        plt.subplot(2,1,1)
        plt.plot(time , data)
        plt.subplot(2,1,2)
        plt.pcolor(X, Y, coefs**2)
        plt.yscale('log')
        plt.show()

    def savePlot(self, data, dt, scales, coefs, fname):
        time = np.arange(0,len(data))*dt
        X, Y = np.meshgrid(time, scales)
        plt.subplot(2,1,1)
        plt.plot(time , data)
        plt.subplot(2,1,2)
        plt.pcolor(X, Y, coefs**2)
        plt.yscale('log')
        plt.savefig(fname)
        plt.close()
