cmake_minimum_required(VERSION 3.20)
project(image_processing_project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(OpenCV_DIR "C:/OpenCV/opencv/build")

find_package(OpenCV REQUIRED)

add_executable(image_processing_project
        static_camera.cpp
        main.cpp
)

target_include_directories(image_processing_project PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(image_processing_project PRIVATE ${OpenCV_LIBS})