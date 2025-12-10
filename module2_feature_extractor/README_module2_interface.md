# Module 1 → Module 2 Audio Interface (v1.0)

- Sample rate: 16,000 Hz (16 kHz)
- Channels: mono
- Frame size from Module 1: 10 ms = 160 samples
- Dtype accepted by Module 2:
  - int16 PCM  (preferred for C++ audio I/O)
  - float32 PCM (range [-1, 1])

Module 2 API:
- Use `StreamingFeatureExtractor`:

  ```python
  from module2_feature_extractor.streaming_feature_extractor import StreamingFeatureExtractor
  import numpy as np

  stream = StreamingFeatureExtractor()

  def on_pcm_frame_from_module1(raw_buffer):
      # Option 1: int16
      pcm = np.frombuffer(raw_buffer, dtype=np.int16)

      # Option 2: if already float32
      # pcm = np.frombuffer(raw_buffer, dtype=np.float32)

      new_mels = stream.process_chunk(pcm)
      # new_mels shape: (80, T_new)
