//
// Created by adeli on 5/17/2025.
//

#ifndef STATIC_CAMERA_H
#define STATIC_CAMERA_H
#include <opencv2/opencv.hpp>
using namespace cv;

Mat captureBackground(const std::string& videoPath, int history = 2000, double learningRate = 0.05);
void processStream(const std::string& videoPath, const Mat& staticBg);

#endif // STATIC_CAMERA_H
