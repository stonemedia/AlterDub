// InputPipeline.h
#pragma once
#include "RingBuffer.h"
#include <vector>
#include <functional>
#include <cmath>

class InputPipeline {
public:
    using FrameCallback = std::function<void(const float*, int)>;

    InputPipeline()
        : m_sampleRate(48000),
          m_frameSize(480),
          m_hopSize(240),
          m_ringBuffer(480 * 4) // default capacity; will update in prepare()
    {
        setInputGainDb(0.0f);
        setupHighPassFilter(80.0f); // default 80 Hz
    }

    // More explicit config function, like a plugin's "prepare to play"
    void prepare(int sampleRate,
                 int frameSizeSamples,
                 int hopSizeSamples,
                 FrameCallback onFrameReady)
    {
        m_sampleRate = sampleRate;
        m_frameSize  = frameSizeSamples;
        m_hopSize    = hopSizeSamples;
        m_onFrameReady = onFrameReady;

        // Resize ring buffer capacity if needed
        // (we recreate it with enough space)
        m_ringBuffer = RingBuffer(static_cast<std::size_t>(m_frameSize * 4));

        // Preallocate temporary buffers
        m_tempBlock.resize(static_cast<std::size_t>(m_frameSize * 4)); // just a safe upper bound
        m_frameBuffer.resize(static_cast<std::size_t>(m_frameSize));

        setInputGainDb(m_inputGainDb); // recompute linear gain
        setupHighPassFilter(m_hpfCutoff);
        reset();
    }

    void reset() {
        // Clear filter states
        m_hpfXPrev = 0.0f;
        m_hpfYPrev = 0.0f;

        // Clear ring buffer
        m_ringBuffer.clear();
    }

    void setInputGainDb(float gainDb) {
        m_inputGainDb = gainDb;
        m_inputGain = std::pow(10.0f, m_inputGainDb / 20.0f);
    }

    void setHighPassEnabled(bool enabled) {
        m_hpfEnabled = enabled;
    }

    void setHighPassCutoff(float cutoffHz) {
        setupHighPassFilter(cutoffHz);
    }

    void setGateEnabled(bool enabled) {
        m_gateEnabled = enabled;
    }

    void setGateThreshold(float threshold) {
        m_gateThreshold = threshold;
    }

    // This is what would be called from the plugin's Process() method
    void processBlock(const float* input, int numSamples) {
        if (!input || numSamples <= 0) {
            return;
        }

        // 1) Apply gain, HPF, gate and write into ring buffer
        // We assume m_tempBlock is large enough (ensured in prepare()).
        for (int i = 0; i < numSamples; ++i) {
            float x = input[i];

            // Input gain
            x *= m_inputGain;

            // High-pass filter (optional)
            if (m_hpfEnabled) {
                x = processHighPass(x);
            }

            // Noise gate (optional)
            if (m_gateEnabled) {
                if (std::fabs(x) < m_gateThreshold) {
                    x = 0.0f;
                }
            }

            m_tempBlock[static_cast<std::size_t>(i)] = x;
        }

        m_ringBuffer.write(m_tempBlock.data(), static_cast<std::size_t>(numSamples));

        // 2) Extract overlapped frames while we can
        processFrames();
    }

private:
    int m_sampleRate{48000};
    int m_frameSize{480}; // e.g. 10 ms at 48 kHz
    int m_hopSize{240};   // e.g. 5 ms hop

    float m_inputGainDb{0.0f};
    float m_inputGain{1.0f};

    // High-pass filter state
    bool  m_hpfEnabled{false};
    float m_hpfCutoff{80.0f};
    float m_hpfAlpha{0.0f};
    float m_hpfXPrev{0.0f};
    float m_hpfYPrev{0.0f};

    // Noise gate state
    bool  m_gateEnabled{false};
    float m_gateThreshold{0.01f}; // small value (~ -40 dBFS)

    RingBuffer m_ringBuffer;
    std::vector<float> m_tempBlock;   // reused for incoming block
    std::vector<float> m_frameBuffer; // reused for outgoing frame
    FrameCallback m_onFrameReady;

    void processFrames() {
        // While we have at least one full frame in the ring buffer
        while (m_ringBuffer.size() >= static_cast<std::size_t>(m_frameSize)) {
            // Peek into preallocated frame buffer
            std::size_t got = m_ringBuffer.peek(
                m_frameBuffer.data(),
                static_cast<std::size_t>(m_frameSize)
            );

            if (got < static_cast<std::size_t>(m_frameSize)) {
                break;
            }

            if (m_onFrameReady) {
                // Call the callback with raw pointer + length
                m_onFrameReady(m_frameBuffer.data(), m_frameSize);
            }

            // Only advance by hopSize to create overlap
            m_ringBuffer.skip(static_cast<std::size_t>(m_hopSize));
        }
    }

    void setupHighPassFilter(float cutoffHz) {
        m_hpfCutoff = cutoffHz;
        float dt = 1.0f / static_cast<float>(m_sampleRate);
        float RC = 1.0f / (2.0f * 3.14159265f * m_hpfCutoff);
        m_hpfAlpha = RC / (RC + dt);

        // Reset filter state
        m_hpfXPrev = 0.0f;
        m_hpfYPrev = 0.0f;
    }

    float processHighPass(float x) {
        // y[n] = alpha * (y[n-1] + x[n] - x[n-1])
        float y = m_hpfAlpha * (m_hpfYPrev + x - m_hpfXPrev);
        m_hpfXPrev = x;
        m_hpfYPrev = y;
        return y;
    }
};
