// Cassini — integrator interface.
#pragma once

#include "cassini/physics/system.hpp"

namespace cassini {

class IIntegrator {
public:
    virtual ~IIntegrator() = default;

    // Advance the system by `dt` days. Implementations may sub-step internally.
    virtual void step(System& sys, double dt) = 0;
};

}  // namespace cassini
