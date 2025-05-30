#ifndef STATIC_CAMERA_H
#define STATIC_CAMERA_H
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
Mat captureBackground(const string& videoPath, int history = 2000, double learningRate = 0.05);
void processStream(const string& videoPath, const Mat& staticBg);

#endif // STATIC_CAMERA_H
