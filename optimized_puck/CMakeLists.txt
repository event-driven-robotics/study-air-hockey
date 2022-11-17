# requires minimum cmake version
cmake_minimum_required(VERSION 3.5)

# produce the cmake var PROJECT_NAME
project(puck_position)

if(NOT CMAKE_BUILD_TYPE)
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY VALUE "Release")
endif()

#include(GNUInstallDirs)

# mandatory use of these packages
find_package(YCM REQUIRED)
find_package(OpenCV REQUIRED)
find_package(YARP COMPONENTS OS sig math dev REQUIRED)
find_package(event-driven REQUIRED)

add_executable(${PROJECT_NAME} main.cpp eros.h tracking.h)

target_link_libraries(${PROJECT_NAME} PRIVATE YARP::YARP_OS
        YARP::YARP_init
        YARP::YARP_sig
        YARP::YARP_dev
        YARP::YARP_math
        ev::event-driven
        ${OpenCV_LIBRARIES})

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_BINDIR})

