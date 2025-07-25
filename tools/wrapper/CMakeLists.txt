cmake_minimum_required(VERSION 3.13)
project(JavaWrapper)

# JNI Wrapper for llama.cpp
find_package(JNI REQUIRED)
find_package(Java REQUIRED)

# Option to build test executable
option(BUILD_WRAPPER_TEST "Build the wrapper test executable" ON)

# Generate the JNI header
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/org_llm_wrapper_LlamaCpp.h
    COMMAND ${Java_JAVAC_EXECUTABLE} -h ${CMAKE_CURRENT_BINARY_DIR} -d ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/org/llm/wrapper/LlamaCpp.java
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/org/llm/wrapper/LlamaCpp.java
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_custom_target(JniHeader ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/org_llm_wrapper_LlamaCpp.h)

# Build JNI wrapper library
add_library(java_wrapper SHARED java_wrapper.cpp)

# Include directories - use paths relative to llama.cpp root
target_include_directories(java_wrapper PRIVATE
    ${JNI_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR} # For the generated JNI header
    ../../include  # llama.cpp include
    ../../common   # common includes
    ../../src      # internal headers if needed
)

add_dependencies(java_wrapper JniHeader)

# Link with llama and common
target_link_libraries(java_wrapper PRIVATE
    llama
    common
)

set_target_properties(java_wrapper PROPERTIES
    OUTPUT_NAME "java_wrapper"
    POSITION_INDEPENDENT_CODE ON
)

# Install JNI library
install(TARGETS java_wrapper LIBRARY)

# Optional test executable
if(BUILD_WRAPPER_TEST)
    add_executable(wrapper_test main.cpp)

    target_include_directories(wrapper_test PRIVATE
        ../../include
        ../../common
    )

    target_link_libraries(wrapper_test PRIVATE
        llama
        common
    )

    install(TARGETS wrapper_test RUNTIME)
endif()

# Compile Java classes
add_custom_command(
    OUTPUT JavaClasses
    COMMAND ${Java_JAVAC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/org/llm/wrapper/*.java
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/org/llm/wrapper/LlamaCpp.java
            ${CMAKE_CURRENT_SOURCE_DIR}/org/llm/wrapper/InitParams.java
            ${CMAKE_CURRENT_SOURCE_DIR}/org/llm/wrapper/GenerateParams.java
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Compiling Java wrapper classes"
)

add_custom_target(java_wrapper_classes ALL DEPENDS JavaClasses)