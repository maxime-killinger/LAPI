cmake_minimum_required(VERSION 3.3)
project(Lapi)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
    lapi.cpp)

add_executable(Lapi ${SOURCE_FILES})

find_package( OpenCV REQUIRED )

target_link_libraries( Lapi ${OpenCV_LIBS} )
