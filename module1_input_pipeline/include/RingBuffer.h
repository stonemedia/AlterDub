// RingBuffer.h
#pragma once
#include <vector>
#include <cstddef>
#include <algorithm>

class RingBuffer {
public:
    explicit RingBuffer(std::size_t capacity)
        : m_buffer(capacity), m_capacity(capacity) {}

    std::size_t capacity() const { return m_capacity; }

    // How many samples are currently stored
    std::size_t size() const { return m_size; }

    // How many samples we can still write
    std::size_t freeSpace() const { return m_capacity - m_size; }

    // Reset buffer state (does not shrink capacity)
    void clear() {
        m_writeIndex = 0;
        m_readIndex  = 0;
        m_size       = 0;
    }

    // Write samples into the buffer. Returns how many were actually written.
    std::size_t write(const float* data, std::size_t count) {
        std::size_t written = 0;
        while (written < count && m_size < m_capacity) {
            m_buffer[m_writeIndex] = data[written];
            m_writeIndex = (m_writeIndex + 1) % m_capacity;
            ++m_size;
            ++written;
        }
        return written;
    }

    // Read samples out of the buffer (and consume them).
    // Returns how many were actually read.
    std::size_t read(float* dest, std::size_t count) {
        std::size_t readCount = 0;
        while (readCount < count && m_size > 0) {
            dest[readCount] = m_buffer[m_readIndex];
            m_readIndex = (m_readIndex + 1) % m_capacity;
            --m_size;
            ++readCount;
        }
        return readCount;
    }

    // Peek samples without consuming them.
    // Returns how many were actually peeked.
    std::size_t peek(float* dest, std::size_t count) const {
        std::size_t available = std::min(count, m_size);
        std::size_t idx = m_readIndex;
        for (std::size_t i = 0; i < available; ++i) {
            dest[i] = m_buffer[idx];
            idx = (idx + 1) % m_capacity;
        }
        return available;
    }

    // Skip (discard) up to 'count' samples without copying them out.
    // Returns how many were actually skipped.
    std::size_t skip(std::size_t count) {
        std::size_t skipped = 0;
        while (skipped < count && m_size > 0) {
            m_readIndex = (m_readIndex + 1) % m_capacity;
            --m_size;
            ++skipped;
        }
        return skipped;
    }

private:
    std::vector<float> m_buffer;
    std::size_t m_capacity{0};
    std::size_t m_writeIndex{0};
    std::size_t m_readIndex{0};
    std::size_t m_size{0};
};
