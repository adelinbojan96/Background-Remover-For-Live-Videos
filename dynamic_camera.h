#ifndef DYNAMIC_CAMERA_H
#define DYNAMIC_CAMERA_H

#include <opencv2/opencv.hpp>
using namespace cv;

void computeGradients(const Mat& prev, const Mat& next, Mat& Ix, Mat& Iy, Mat& It);

bool isSimilarColor(const Vec3b& color, const std::vector<Vec3b>& colors, int tol=300);

Mat hornSchunck(const Mat& prev, const Mat& next, float alpha = 1.0f, int numIter = 100);

Point2f estimateGlobalTranslation(const Mat& flow);

Point2f estimateGlobalTranslationRANSAC(const Mat& prevGray, const Mat& gray);

Mat compensateTranslation(const Mat& frame, const Point2f& t);

Mat detectMovingMask(const Mat& flow, float thresh = 1.0f);

Mat removeMovingObjects(const Mat& frame, const Mat& mask, int inpaint_radius = 10);

Mat dimMaskRegion(const Mat& frame, const Mat& mask, float alpha = 0.5f);

Mat gaussianBlur(const Mat& src, int ksize, float sigma);

void dilate(const Mat& src, Mat& dst, const Mat& kernel);

Mat ellipseKernel_2(int ksize);

#endif
