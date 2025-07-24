# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# flags

llama_add_compile_flags()

# tools

if (EMSCRIPTEN)
else()
    # Only build JNI wrapper - other tools were removed from this fork
    add_subdirectory(wrapper)
endif()
