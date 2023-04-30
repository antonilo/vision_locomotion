project(utils)
cmake_minimum_required(VERSION 2.8.3)
message("Detected cmake version ${CMAKE_VERSION}")

find_package(catkin_simple REQUIRED)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_definitions(-std=c++17)

catkin_simple()


cs_add_library(logging src/logging.cpp)
target_link_libraries(logging stdc++fs
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS})


cs_install()

cs_export()
