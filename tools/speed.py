import numpy as np
import time

# Warm-up
a = np.random.rand(1024)
np.fft.fft(a)

# Benchmark
start = time.time()
for _ in range(2000):
    np.fft.fft(a)
end = time.time()

print("Time for 2000 FFTs:", end - start, "seconds")
print("Per FFT:", (end - start)/2000, "sec")
