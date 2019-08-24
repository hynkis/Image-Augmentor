cmake_minimum_required(VERSION 3.5.1)
project(image_augmentor)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED COMPONENTS system)

# Find Opencv
find_package(OpenCV REQUIRED)

# Find OpenMP, parallel processing
# find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${Boost_INCLUDE_DIRS}
  #${OpenCV_INCLUDE_DIRS} # INCLUDE_DIRS returns directory.
  ${OpenCV_LIBRARIES} # LIBRARIES returns lib.
)

add_executable(image_augmentor src/augmentor.cpp)
target_link_libraries(image_augmentor ${OpenCV_LIBRARIES}) # should be lib, not directory.
