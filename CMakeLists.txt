cmake_minimum_required(VERSION 3.20)
project(MLProject LANGUAGES CXX C)

# Set C++ standard and project-wide settings
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# Configure compiler options
add_compile_options(
    "$<$<C_COMPILER_ID:MSVC>:/utf-8>"
    "$<$<CXX_COMPILER_ID:MSVC>:/utf-8>"
)

# Include directories
target_include_directories(MLProject PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/Algorithms
    ${CMAKE_CURRENT_SOURCE_DIR}/DeepLearning
)

# Source files (explicit listing matching your vcxproj)
set(SOURCES
    Algorithms/Kmeans.cpp
    Algorithms/Knn.cpp
    common.cpp
    data.cc
    DataHandler.cc
    DeepLearning/Layer.cpp
    DeepLearning/Network.cpp
    DeepLearning/Neuron.cpp
    Source.cpp
)

# Headers (for IDE organization)
set(HEADERS
    Algorithms/Kmeans.h
    common.h
    DataConfig.hpp
    Algorithms/Knn.h
    data.hpp
    DataHandler.hpp
    DeepLearning/HiddenLayer.h
    DeepLearning/InputLayer.h
    DeepLearning/Layer.h
    DeepLearning/Network.h
    DeepLearning/Neuron.h
    DeepLearning/OutputLayer.h
)

# Create executable
add_executable(MLProject ${SOURCES} ${HEADERS})

# Compiler definitions
target_compile_definitions(MLProject PRIVATE
    $<$<CONFIG:Debug>:_DEBUG>
    $<$<PLATFORM_ID:Windows>:UNICODE _UNICODE>
    _CONSOLE
)

# Platform-specific settings
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    # Linker settings
    target_link_options(MLProject PRIVATE
        /SUBSYSTEM:CONSOLE
        $<$<CONFIG:Release>:
            /OPT:REF
            /OPT:ICF
            /LTCG
        >
    )

    # Compiler flags
    target_compile_options(MLProject PRIVATE
        /W3
        /sdl
        /permissive-
        $<$<CONFIG:Release>:
            /GL
            /Gw
            /Gy
        >
    )
endif()

# Data file handling
add_custom_command(TARGET MLProject POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    "${CMAKE_SOURCE_DIR}/train-images-idx3-ubyte"
    $<TARGET_FILE_DIR:MLProject>
)

add_custom_command(TARGET MLProject POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    "${CMAKE_SOURCE_DIR}/train-labels-idx1-ubyte"
    $<TARGET_FILE_DIR:MLProject>
)

# Configure output
set_target_properties(MLProject PROPERTIES
    OUTPUT_NAME "MLProject"
    SUFFIX ".exe"
    VS_PROJECT_NAME "MLProject"
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>"
)

# Enable parallel builds for MSVC
if(CMAKE_GENERATOR MATCHES "Visual Studio")
    set_property(TARGET MLProject PROPERTY VS_GLOBAL_EnableMP "true")
endif()