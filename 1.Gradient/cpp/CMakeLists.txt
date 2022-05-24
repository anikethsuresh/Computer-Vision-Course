# Set the minimum version for cmake
cmake_minimum_required(VERSION 3.5)
# Name of the project
project(Gradient)

# Set some flags and vars
# set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -std=c++17")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fsanitize=address -std=c++17")
set (source_dir "${PROJECT_SOURCE_DIR}/src/")

# Update the CMAKE_PREFIX_PATH so we can find OpenCV
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};../../downloaded-libraries/libraries/opencv/build/")
find_package(OpenCV REQUIRED )

# Add a target "stats" that is built from the source files included
add_executable (gradient "${source_dir}main.cpp")
# Add libraries to the stats target
target_link_libraries(gradient PUBLIC ${OpenCV_LIBS} )
