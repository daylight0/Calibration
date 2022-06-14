cmake_minimum_required(VERSION 3.0.0)
project(calibrate VERSION 0.1.0)

include(CTest)
enable_testing()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)

# 设置编译模式
set(CMAKE_BUILD_TYPE "Debug")

find_package(OpenCV 4.5.5 REQUIRED)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        "/usr/local/include/opencv4/"
        "/usr/local/include/opencv4/opencv2/"
)

add_executable(main main.cpp)
target_link_libraries(main ${OpenCV_LIBS})

add_executable(stereo stereo.cpp)
target_link_libraries(stereo ${OpenCV_LIBS})

add_executable(mono mono.cpp)
target_link_libraries(mono ${OpenCV_LIBS})

add_executable(monocali monoCalibration.cpp)
target_link_libraries(monocali ${OpenCV_LIBS})

add_executable(stereoCalibration stereoCalibration.cpp)
target_link_libraries(stereoCalibration ${OpenCV_LIBS})

set (EXECUTABLE_OUTPUT_PATH /home/wh/slam/Calibration/bin) # 设置了可执行文件的输出路径，其中的大写表示cmake中的变量
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
