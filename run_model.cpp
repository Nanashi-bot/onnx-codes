#include <iostream>
#include <vector>
#include <onnxruntime_cxx_api.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

int main() {
    // Initialize onnx runtime environment and session
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "model");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    Ort::Session session(env, "model.onnx", session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    // stb_image used to load image
    int width, height, channels;
    unsigned char* img = stbi_load("input.jpg", &width, &height, &channels, 3);
    if (!img) {
        std::cerr << "Failed to load image\n";
        return -1;
    }

    // TODO: NEED TO AUTOMATICALLY RESIZE IMAGE TO 224x224

    // Convert image data to float and normalize
    const int img_size = 224 * 224 * 3;
    std::vector<float> input_tensor_values(img_size);
    for (int i = 0; i < 224 * 224; i++) {
        input_tensor_values[i] = (img[3*i] / 255.0f - 0.485f) / 0.229f;        // R
        input_tensor_values[i + 224*224] = (img[3*i + 1] / 255.0f - 0.456f) / 0.224f; // G
        input_tensor_values[i + 2*224*224] = (img[3*i + 2] / 255.0f - 0.406f) / 0.225f; // B
    }
    stbi_image_free(img);

    // input shape: {batch_size, channels, height, width}
    std::array<int64_t,4> input_shape{1, 3, 224, 224};

    // Create input tensor object from data values
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

    // Get input and output node names
    std::vector<std::string> input_names = session.GetInputNames();
    std::vector<std::string> output_names = session.GetOutputNames();

    // Initialize input and output name pointers
    std::vector<const char*> input_names_ptr;
    std::vector<const char*> output_names_ptr;
    for (const auto& s : input_names) input_names_ptr.push_back(s.c_str());
    for (const auto& s : output_names) output_names_ptr.push_back(s.c_str());

    // Inference
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names_ptr.data(), &input_tensor, 1, output_names_ptr.data(), 1);

    // Pointer points to output tensor float values
    float* floatarr = output_tensors.front().GetTensorMutableData<float>();
    int output_size = output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount();

    // Find index of max output value
    int max_idx = 0;
    float max_val = floatarr[0];
    for (int i = 1; i < output_size; i++) {
        if (floatarr[i] > max_val) {
            max_val = floatarr[i];
            max_idx = i;
        }
    }

    std::cout << "Predicted class index: " << max_idx << "\n";

    return 0;
}

