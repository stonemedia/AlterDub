import numpy as np
import matplotlib.pyplot as plt

# Use raw string or forward slashes
path = r"module2_feature_extractor\data\features\spk001\demo_logmel.npy"
# OR use forward slashes
# path = "module2_feature_extractor/data/features/spk001/demo_logmel.npy"

# Load the log-mel matrix
logmel = np.load(path)

print("Shape:", logmel.shape)

plt.figure(figsize=(10, 4))
plt.imshow(logmel, aspect='auto', origin='lower')
plt.colorbar()
plt.title("Log-Mel Spectrogram")
plt.xlabel("Frames")
plt.ylabel("Mel Channels (80)")
plt.show()
