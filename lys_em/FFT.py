import pyfftw

fft, ifft = pyfftw.interfaces.numpy_fft.fft2, pyfftw.interfaces.numpy_fft.ifft2
pyfftw.interfaces.cache.enable()
