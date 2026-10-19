// Cassini — library version macros.
#pragma once

#define CASSINI_VERSION_MAJOR 0
#define CASSINI_VERSION_MINOR 1
#define CASSINI_VERSION_PATCH 0
#define CASSINI_VERSION_STRING "0.1.0"

namespace cassini {
inline constexpr int version_major = CASSINI_VERSION_MAJOR;
inline constexpr int version_minor = CASSINI_VERSION_MINOR;
inline constexpr int version_patch = CASSINI_VERSION_PATCH;
}  // namespace cassini
