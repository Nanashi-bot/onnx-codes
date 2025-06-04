cmake_minimum_required(VERSION 3.10)
project(RunModel)

set(CMAKE_CXX_STANDARD 17)

set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/onnxruntime-linux-x64-1.22.0")

include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

add_executable(run_model run_model.cpp)
target_link_libraries(run_model onnxruntime ${OpenCV_LIBS})

