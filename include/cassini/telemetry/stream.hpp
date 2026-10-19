// Cassini — live telemetry streaming interface.
//
// Dependency-free: defines a frame format and a sink abstraction that a gRPC
// or WebSocket transport can implement. Frames are the same body layout as the
// binary snapshot (bit-exact), plus a monotonic frame counter for the viewer.
#pragma once

#include <cstdint>
#include <vector>

#include "cassini/data/snapshot.hpp"
#include "cassini/physics/system.hpp"

namespace cassini::telemetry {

// A sink consumes serialized telemetry frames (e.g., writes to a socket).
class TelemetrySink {
public:
    virtual ~TelemetrySink() = default;
    virtual void send(const std::vector<std::uint8_t>& frame) = 0;
};

class TelemetryStream {
public:
    explicit TelemetryStream(TelemetrySink& sink) : sink_(sink) {}

    // Emit a frame describing the current system state.
    void emit(const cassini::System& sys) {
        std::vector<std::uint8_t> body;
        data::write_snapshot(sys, body);
        frame_.clear();
        frame_.push_back(static_cast<std::uint8_t>(counter_ >> 0));
        frame_.push_back(static_cast<std::uint8_t>(counter_ >> 8));
        frame_.push_back(static_cast<std::uint8_t>(counter_ >> 16));
        frame_.push_back(static_cast<std::uint8_t>(counter_ >> 24));
        frame_.insert(frame_.end(), body.begin(), body.end());
        sink_.send(frame_);
        ++counter_;
    }

private:
    TelemetrySink& sink_;
    std::uint32_t counter_{0};
    std::vector<std::uint8_t> frame_;
};

}  // namespace cassini::telemetry
