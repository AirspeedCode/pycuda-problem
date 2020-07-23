import tensorflow as tf
import tensorflow.contrib.tensorrt as trt # must import this although we will not use it explicitly
from tensorflow.python.platform import gfile

import os
import pickle
import numpy as np
import time

from scipy.io import wavfile
import math
import scipy.signal
from scipy.fftpack import fft
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import cv2

# function to read a ".pb" model 
# (can be used to read frozen model or TensorRT model)
def read_pb_graph(model):
  with gfile.FastGFile(model,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  return graph_def

def load_audio(path):
    try:
        #print(path)
        sample_rate, raw_data = wavfile.read(path)
        #print(len(raw_data))
        return raw_data, sample_rate
    except Exception as e:
        print("load_audio:", e)
        pass

def preEmphasis(signalIn, factor):
    # A pre-emphasis filter is useful in several ways: (1) balance the frequency spectrum since
    # high frequencies usually have smaller magnitudes compared to lower frequencies, (2) avoid
    # numerical problems during the Fourier transform operation and (3) may also improve the
    # Signal-to-Noise Ratio (SNR).
    # see https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html

    try:
        #signalOut = np.zeros(len(signalIn), dtype=type(signalIn[0]))
        signalOut = np.zeros(len(signalIn), dtype=float)
        for t in range(len(signalIn)):  # sweep through array
            if t == 0:
                signalOut[t] = 0
            else:
                signalOut[t] = signalIn[t] - (factor * signalIn[t - 1])
        return signalOut
    except Exception as e:
        print("preEmphasis: An exception occurred",e)
        pass

def mel(f):
    """
    Mel function - converts a frequency number in Hz into a Mel scale log value

    Args:
        f : frequency in Hz

    Returns:
        m : mel scale
    """
    return 1125.0 * np.log(1.0 + f / 700.0)

def mel_inv(m):
    """
    Inverse Mel function - converts a Mel-scale frequency number into Hz

    Args:
        m : mel scale

    Returns:
        f : frequency in Hz
    """
    return 700.0 * (np.exp(m / 1125.0) - 1.0)

def mel_freq_fbank_weight(n, freq, fs, fmax, fmin=0.0):
    """
    Generates Mel-freqency filter bank weights

    Args:
        n    : number of filter banks
        freq : center of frequency bins as discrete value (-0.5 ~ 0.5),
               can be computed by numpy.fft.fftfreq
        fs   : sample rate
        fmax : maximal frequency in Hz
        fmin : (default 0) minimal frequency in Hz

    Returns:
        fbw  : filter bank weights, indexed by 'bf'.
               'b' is the index of filter bank.
    """
    mmax = mel(fmax)  # maximum frequency on the mel scale
    mmin = mel(fmin)  # mimimum frequency on the mel scale
    mls = np.linspace(mmin, mmax, n + 2)  # an array of 42 evenly distributed numbers on the mel scale (n=40)
    fls = mel_inv(mls)  # an array of 42 evenly distributed numbers on the Hz scale (n=40) between 0 hz an 22050 hz
    fbw = np.zeros((n, len(freq)))  # 2D array (40 x 40) (number of filter banks, centre of frequency bins)
    freq = np.abs(fs * freq)

    # per bank
    for i in range(n):
        # left slope
        left = (freq - fls[i]) / (fls[i + 1] - fls[i])
        left[left < 0.0] = 0.0
        left[left > 1.0] = 0.0
        # right slope
        right = (fls[i + 2] - freq) / (fls[i + 2] - fls[i + 1])
        right[right < 0.0] = 0.0
        right[right >= 1.0] = 0.0
        # sum
        fbw[i] = left + right

    assert np.min(fbw) == 0.0
    assert np.max(fbw) <= 1.0
    return fbw
    
def hamming(data):
    a0 = 0.53836
    a1 = 1 - a0
    length = len(data)
    x = np.linspace(0, length - 1, length)
    window = a0 - a1 * np.cos((2 * math.pi * x) / length)
    return data * window

def get_ffts(frameData, NFFT = 1024):
    """
    frameData : Input, 2D array of audio samples. Index is (channel, sample). Typical shape = (2, 4410)
    NFFT      : Input, number of bins for FFT, defaults to 1024
    cf        : Output, 2D array of audio spectra. Index is (channel, bin),
                    i.e. (2 channels, 4096 frequency bins, etc)
    """
    nchannels = len(frameData[:,0])
    cf = np.zeros([nchannels, NFFT], dtype = np.complex128)
    
    for channel in range(nchannels):
        cf[channel,:] = fft(x=frameData[channel,:], n=NFFT)
    
    return cf

def get_tau(theta, temperature, sample_rate):
    aperture = 0.5
    c = 331.3 + 0.606 * temperature
    tau = int ((aperture * math.sin(math.radians(theta)) * sample_rate) / c)
    return tau

def cross_power_spectral_density(cf, fw, NFFT):
    
    """Cross Power Spectral Density

    Args:
        cf  : multi-channel frequency domain signal, indices (cf) - channel, frequency
        fw  : (default 2) half width of neighbor area in freq domain,
              including center
    Returns:
        cpsd: cross power spectral density
    """
    
    # covariance matrix
    newCov = np.einsum('cf,df->cdf', cf, cf.conj())
    temp2 = newCov[0, 1, :]
    
    # kernal for convolution
    newKernel = np.hanning(fw * 2 + 1)[1:-1]
    newKernel = newKernel / np.sum(newKernel)    # normalize, Array of shape (1, 15)

    _apply_conv = scipy.ndimage.filters.convolve

    i = 0
    j = 1

    cpsd3 = np.zeros((NFFT,), dtype=np.complex128)

    rpart = _apply_conv(newCov[i,j,:].real, newKernel, mode='nearest')
    ipart = _apply_conv(newCov[i,j,:].imag, newKernel, mode='nearest')

    cpsd3 = rpart + 1j * ipart
    return cpsd3

mod = SourceModule("""
    #include <pycuda-complex.hpp>
    #include <stdio.h>

    typedef pycuda::complex<float> cmplx;
    __global__ void cc_gpu(float *fbwn, cmplx *cpsd_phat, cmplx *steer, float *cc, int BINS, int WINDOW, int BANKS)
    {

        // fbwn      : input, array shape (40, 4096), of type float32
        // cpsd_phat : input, vector shape (4096,), of type complex64
        // steer     : input, array shape (4096, 121), of type complex64
        // cc        : output, array shape (40, 121), of type float32

        int x = threadIdx.x + blockIdx.x * blockDim.x;     
        int y = threadIdx.y + blockIdx.y * blockDim.y;

        int WIDTH = blockDim.x * gridDim.x;
        int HEIGHT = blockDim.y * gridDim.y;

        int XM = (x + WIDTH) % WIDTH;
        int YM = (y + HEIGHT) % HEIGHT;

        int INDEX = XM + YM * WIDTH; // CUDA uses pseudo-mutimensional array indexing (i.e. it's a 1D flat array)

        int bank = y;
        int delay = x;

        float Pvalue = 0.f; 

        if(x < WINDOW && y < BANKS)
        {  
            for (int bin = 0; bin < BINS; bin++)
            {
                if (fbwn[bank * BINS + bin] > 0)
                {
                    //Pvalue += fbwn[y,f] * (cpsd_phat[f] * steer[f,x]).real(); // Python equivalent code for reference                                        
                    Pvalue += (fbwn[bank * BINS + bin] * (cpsd_phat[bin] * steer[bin * WINDOW + delay]).real());
                }
            }
        }  
        cc[INDEX] = Pvalue; 
    }
    """)

def gcc_phat_fbanks_gpu3(cpsd, fbwn_gpu, steer_gpu, ccfb_gpu, zoom, eps):
    
    """GCC-PHAT on filter banks

    Args:
        cpsd  : 1D array of length 'nbins' of Cross Power Spectral Density, i.e. a measure of 
                both the power shared by a given frequency for two signals, and the phase 
                shift between the two signals at that frequency. Each element is of type complex128. 
                Typical shape is (4096,) of complex numbers if the FFT has 4096 bins.
                
        fbwn  : 2D array containing Normalised filter bank weights, indexed by (filter bank, frequency bin)
                The sum of values for each filter bank = 1.0, although their frequency distribution 
                can vary between banks. Typical shape is (40, 4096) of float64 elements. 
                   
        steer : 2D array of steering factors. Indexed by (frequency bin, sample delay). 
                Each element is of type complex128. Typical shape is (4096, 121) for +/- 60 sample 
                delays and 4096 frequency bins
                
        zoom  : constant used to dimension sample delay axis. E.g zoom = 60 gives a range of +/- 60 sample delays

        eps   : (default 0.0) small constant added to the CPSD_PHAT denominator for
                numerical stability, as well as to suppress low engergy
                bins.
    Return:
        fbcc : 2D array of GCC-PHAT on filter banks. Index is (filter bank, sample delay). 
               Typical shape = (40, 121). Each element is type float64, 
               normalised to between -1.0 and 1.0
               
    """
    WINDOW = np.int32((2 * zoom) + 1)
    BANKS = np.int32(len(fbwn_gpu[:,0]))
    BINS = np.int32(len(cpsd))
      
    cpsd_phat = ( cpsd / (np.abs(cpsd) + eps) ).astype(np.complex64) #Array shape (4096,)
    cpsd_phat_gpu = gpuarray.to_gpu(cpsd_phat)
    
    BLOCK_X = 32 # Number of threads in block X direction (X * Y must be < 1025)
    BLOCK_Y = 32 # Number of threads in block Y direction (X * Y must be < 1025)
    GRID_X = math.ceil(WINDOW / BLOCK_X) # Required number of blocks in X direction of grid
    GRID_Y = math.ceil(BANKS / BLOCK_Y)  # Required number of blocks in Y direction of grid
         
    func = mod.get_function("cc_gpu")
    func(fbwn_gpu, cpsd_phat_gpu, steer_gpu, ccfb_gpu, BINS, WINDOW, BANKS,
         block=(BLOCK_X, BLOCK_Y, 1), grid=(GRID_X, GRID_Y, 1)
         )

    return ccfb_gpu.get()[:BANKS,:WINDOW]

def init_gcc_phat_fbanks_gpu(zoom, nbanks, sampleRate, NFFT, maxFreq = None,  
                             freq = None):
    """ 
    Initialise the GCC PHAT on filter banks algorithm

    Args:
    
        zoom  : constant used to dimension sample delay axis. 
                E.g zoom = 60 gives a range of +/- 60 sample delays
        
        nbanks : Number of filter banks (i.e. 40)
        
        sampleRate : Sample rate of the audio clip in samples per second (i.e.44100)
        
        maxFreq : Maximum frequency range of the algorithm. defaults to Sample Rate /2
        
        NFFT  : Number of bins used for FFT (e.g. 4096)
        
        freq  : Center of frequency bins as discrete value (-0.5 ~ 0.5). By 
                default it is computed by numpy.fft.fftfreq with fft size. 1D 
                array. Shape is (4096,) of type float64
        
    Return:
      
        freq  : Center of frequency bins as discrete value (-0.5 ~ 0.5). By 
                default it is computed by numpy.fft.fftfreq with fft size. 1D 
                array. Shape is (4096,) of type float64
        
        steer : 2D array of steering factors. Indexed by (frequency bin, sample 
                delay). Each element is of type complex128. Typical shape is 
                (4096, 121) for +/- 60 sample delays and 4096 frequency bins
        
        fbwn  : 2D array containing Normalised filter bank weights, indexed by 
                (filter bank, frequency bin). The sum of values for each filter 
                bank = 1.0, although their frequency distribution can vary 
                between banks. Typical shape is (40, 4096) of float64 elements. 
    """
    WINDOW = np.int32((2 * zoom) + 1)
    BANKS = np.int32(nbanks)
    
    BLOCK_X = 32 # Number of threads in block X direction (X * Y must be < 1025)
    BLOCK_Y = 32 # Number of threads in block Y direction (X * Y must be < 1025)
    GRID_X = math.ceil(WINDOW / BLOCK_X) # Required number of blocks in X direction of grid
    GRID_Y = math.ceil(BANKS / BLOCK_Y)  # Required number of blocks in Y direction of grid
    WIDTH = BLOCK_X * GRID_X # Number of columns of threads in array
    HEIGHT = BLOCK_Y * GRID_Y # Number of rows of threads in array
    
    if freq is None:
            freq = np.fft.fftfreq(NFFT)
                
    if maxFreq is None:
            maxFreq = int(sampleRate / 2)
            
    delay = np.arange(-zoom, zoom + 1, dtype=float)
    steer = np.exp(-2j * math.pi * np.outer(freq, delay))
  
    # generate the filterbanks
    fbw = mel_freq_fbank_weight(n = nbanks, 
                                freq = freq, 
                                fs = sampleRate, 
                                fmax = maxFreq, 
                                fmin=0.0)
    
    # normalize the filter bank weights. The sum of values for each filter 
    # bank = 1.0, although their frequency distribution can vary between banks.
    
    fbwn = fbw / np.sum(fbw, axis=1, keepdims=True) 
    
    # Do this during initialisation, as it doesn't change
    fbwn_gpu = gpuarray.to_gpu(fbwn.astype(np.float32))
    steer_gpu = gpuarray.to_gpu(steer.astype(np.complex64))
    ccfb = np.zeros((HEIGHT,WIDTH), dtype = np.float32) # needs to be single precision to run on CUDA (i.e. 32 bit)
    ccfb_gpu = gpuarray.to_gpu(ccfb)
    
    #return freq, delay, steer, fbwn
    return freq, steer_gpu, fbwn_gpu, ccfb_gpu

def init_GCCFB(nBanks, sample_rate):
    
    # nBanks : Number of filter banks

    MIN_TEMPERATURE = -32 # Minimum operating temperature, per MIL-STD-810G, climatic category 'C1'- 'Basic Cold'
    MAX_THETA       = 60  # maximum value of theta in degrees. MAX_THETA = 60 provides window of +/- 60 degree sector

    # Constants
    freqWidth = 4
    eps = 1e-15

    max_tau = get_tau(theta = MAX_THETA, temperature = MIN_TEMPERATURE, sample_rate = sample_rate) # maximum sample delay based on lowest speed of sound
    zoom = int(math.ceil(max_tau / 10.0)) * 10 # round 'zoom' requirement up to nearest then samples
    maxFreq = sample_rate // 2
    freq, steer_gpu, fbwn_gpu, ccfb_gpu = init_gcc_phat_fbanks_gpu(zoom,
                                                                               nbanks=nBanks,
                                                                               sampleRate=sample_rate,
                                                                               maxFreq=maxFreq,
                                                                               #NFFT=c.NFFT,
                                                                               NFFT = 4096,
                                                                               freq=None)
    return freq, steer_gpu, fbwn_gpu, ccfb_gpu, freqWidth, zoom, eps

def get_GCCFB(path, freq, steer_gpu, fbwn_gpu, ccfb_gpu, freqWidth, zoom, eps):  
    #try:
    
    # load the wav file into memory
    audio_data, sample_rate = load_audio(path)

    #Apply pre-emphasis
    PRE_EMP = 0.97
    left_signal = preEmphasis(signalIn = audio_data[:,0], factor = PRE_EMP)
    right_signal = preEmphasis(signalIn = audio_data[:,1], factor = PRE_EMP)

    #plot the audio data
    #plot_audio(left_data = left_signal, right_data = right_signal)

    #Generate the spectrum
    #tempFrame = np.zeros([c.NCHANNELS, c.BUFFER_LENGTH], dtype=np.int32)
    tempFrame = np.zeros([2, 4800], dtype=np.int32)
    tempFrame[0,:] = hamming(left_signal) # Also switch to correct left /right error
    tempFrame[1,:] = hamming(right_signal)
    #cf = get_ffts(frameData=tempFrame, NFFT=c.NFFT)
    cf = get_ffts(frameData=tempFrame, NFFT=4096)
    #plot_spectrum(sample_rate = sample_rate, NFFT= NFFT, cf = cf)

    #Generate GCCFB
    #new_cpsd = cross_power_spectral_density(cf=cf, fw=freqWidth, NFFT=c.NFFT)
    new_cpsd = cross_power_spectral_density(cf=cf, fw=freqWidth, NFFT=4096)
    gcc_data = gcc_phat_fbanks_gpu3(new_cpsd, fbwn_gpu, steer_gpu, ccfb_gpu, zoom, eps)

    #Normalise data to range 0.0 to 1.0
    gcc_data = (gcc_data + 1)/2

    #Conver to 8-bit image
    img_arr = (gcc_data * 255).astype('uint8')
    img_arr = cv2.flip(img_arr, 0 )

    #except Exception as e:
    #    print("get_GCCFB:", e)
    #    pass
    
    return img_arr


# Initialise GCCFB algo
freq, steer_gpu, fbwn_gpu, ccfb_gpu, freqWidth, zoom, eps = init_GCCFB(nBanks = 40, sample_rate = 48000)

full_path = '20-06-16-226731400-PH3P-022-041-2984.wav'
trt_model = 'mantis_TensorRT_model.pb'

#**** get_GCCFB works when called outside the tensorflow session, but not inside it ****
# Uncomment the following two lines to see output of get_GCCFB
img_arr = get_GCCFB(full_path, freq, steer_gpu, fbwn_gpu, ccfb_gpu, freqWidth, zoom, eps)
print(img_arr)

# open a tensorRT session
graph = tf.Graph()
with graph.as_default():
    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.50))) as sess:
        
        # read TensorRT model
        trt_graph = read_pb_graph(trt_model)

        # obtain the corresponding input-output tensor
        tf.import_graph_def(trt_graph, name='')
        input = sess.graph.get_tensor_by_name('conv2d_input:0')
        output = sess.graph.get_tensor_by_name('dense/Sigmoid:0')
        
        img_arr = get_GCCFB(full_path, freq, steer_gpu, fbwn_gpu, ccfb_gpu, freqWidth, zoom, eps)
        img_arr = img_arr/255
        input_img = img_arr.reshape( (1, 40, 141, 1) )     
        out_pred = sess.run(output, feed_dict={input: input_img})
        sess.close()

print(input_img)
print(out_pred)