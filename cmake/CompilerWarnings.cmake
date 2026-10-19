# Zero-warning build policy: -Wall -Wextra -Wpedantic -Werror (GCC/Clang),
# /W4 /WX (MSVC), applied project-wide via add_compile_options.
#
# C4127 ("conditional expression is constant") is disabled on MSVC because it
# is spuriously triggered by assertion macros with constant conditions.

function(enable_project_warnings)
  set(msvc_warnings
      /W4
      /permissive-
      /WX
      /wd4127
      /w14242 /w14254 /w14263 /w14265 /w14287 /w14296 /w14545 /w14546 /w14547
      /w14826 /w14905 /w14906 /w14928)

  set(clang_warnings
      -Wall
      -Wextra
      -Wpedantic
      -Werror
      -Wshadow
      -Wconversion
      -Wsign-conversion
      -Wdouble-promotion
      -Wnull-dereference)

  set(gcc_warnings
      -Wall
      -Wextra
      -Wpedantic
      -Werror
      -Wshadow
      -Wconversion
      -Wsign-conversion
      -Wdouble-promotion
      -Wnull-dereference
      -Wformat=2
      -Wundef)

  if(MSVC)
    set(warnings ${msvc_warnings})
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(warnings ${clang_warnings})
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(warnings ${gcc_warnings})
  else()
    message(AUTHOR_WARNING "Unknown compiler; no warnings configured.")
  endif()

  # Apply to CXX only (not CUDA, whose VS integration mangles generator
  # expressions).
  string(JOIN " " warnings_str ${warnings})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${warnings_str}" PARENT_SCOPE)
endfunction()
