// main.cpp
#include "InputPipeline.h"
#include <iostream>
#include <cmath>
#include <algorithm>

int main() {
    const int sampleRate = 48000;
    const int frameSize = 480;    // 10 ms frames
    const int hopSize   = 240;    // 5 ms hop (50% overlap)
    const int blockSize = 64;     // pretend Pro Tools buffer

    int frameCounter = 0;

    InputPipeline pipeline;

    // Define what happens when a frame is ready
    InputPipeline::FrameCallback callback =
        [&](const float* frameData, int frameLen) {
            ++frameCounter;
            std::cout << "Frame " << frameCounter
                      << " ready, size = " << frameLen << "\n";
        };

    // Prepare the pipeline (like a plugin "prepare to play")
    pipeline.prepare(sampleRate, frameSize, hopSize, callback);

    // Optional: tweak settings
    pipeline.setInputGainDb(0.0f);       // 0 dB gain
    pipeline.setHighPassEnabled(true);   // enable HPF at 80 Hz
    pipeline.setGateEnabled(true);       // enable gate
    pipeline.setGateThreshold(0.001f);   // low threshold so we don't kill the sine

    // Generate 1 second of a 440 Hz sine wave
    const int totalSamples = sampleRate; // 1 second
    std::vector<float> fakeAudio(totalSamples);

    const float freq = 440.0f;
    for (int n = 0; n < totalSamples; ++n) {
        float t = static_cast<float>(n) / sampleRate;
        fakeAudio[n] = std::sin(2.0f * 3.14159265f * freq * t);
    }

    // Feed in blocks, like a host would
    int index = 0;
    while (index < totalSamples) {
        int remaining = totalSamples - index;
        int thisBlock = std::min(blockSize, remaining);

        pipeline.processBlock(fakeAudio.data() + index, thisBlock);
        index += thisBlock;
    }

    std::cout << "Done. Total frames: " << frameCounter << std::endl;

    // Demonstrate reset (optional)
    pipeline.reset();

    return 0;
}
