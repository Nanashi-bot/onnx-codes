# ONNX Model Inference with C++

This repository demonstrates running inference on ONNX models using C++ and the ONNX Runtime.

## Structure

onnx/
├── CMakeLists.txt
├── run_model.cpp
├── stb_image.h
├── models/
│ ├── model.onnx # Place your ONNX model(s) here
│ └── input.jpg # Test input image
├── build/ # Created during compilation (ignored)
└── README.md


## Requirements

- CMake
- C++17 compiler
- ONNX Runtime (prebuilt)

## Setup

1. Download the ONNX Runtime prebuilt package for Linux:
   https://github.com/microsoft/onnxruntime/releases

2. Extract it and place it in the root of the repo:

onnxruntime-linux-x64-1.22.0/


3. Build the project:
```bash
mkdir build
cd build
cmake ..
make
```

## Running

Run the compiled executable with a test input:

./run_model

Make sure model.onnx and input.jpg are present in the same directory as run_model.

## Notes

run_model.cpp handles loading the model and running inference.

stb_image.h is used for image loading.
    
    


