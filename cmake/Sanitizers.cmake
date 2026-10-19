# ASan / UBSan support (GCC & Clang only; MSVC has no equivalent flag set here).
# TSan is exercised separately in CI to avoid runtime conflicts with ASan.
if(CASSINI_ENABLE_SANITIZERS)
  if(NOT MSVC)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
  else()
    message(WARNING "CASSINI_ENABLE_SANITIZERS has no effect under MSVC.")
  endif()
endif()
