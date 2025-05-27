#include "dynamic_camera.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>

using namespace cv;

void computeGradients(const Mat& prev, const Mat& next, Mat& Ix, Mat& Iy, Mat& It) {
    Mat prevF, nextF;
    prev.convertTo(prevF, CV_32F);
    next.convertTo(nextF, CV_32F);
    Sobel(prevF, Ix, CV_32F, 1, 0, 3);
    Sobel(prevF, Iy, CV_32F, 0, 1, 3);
    It = nextF - prevF;
}

Mat hornSchunck(const Mat& prev, const Mat& next, float alpha, int numIter) {
    Mat Ix, Iy, It;
    computeGradients(prev, next, Ix, Iy, It);

    Mat u = Mat::zeros(prev.size(), CV_32F);
    Mat v = Mat::zeros(prev.size(), CV_32F);
    Mat uAvg, vAvg;
    for (int k = 0; k < numIter; ++k) {
        blur(u, uAvg, Size(3,3));
        blur(v, vAvg, Size(3,3));
        Mat denom = alpha*alpha + Ix.mul(Ix) + Iy.mul(Iy);
        Mat term  = Ix.mul(uAvg) + Iy.mul(vAvg) + It;
        u = uAvg - Ix.mul(term) / denom;
        v = vAvg - Iy.mul(term) / denom;
    }
    Mat flow(prev.size(), CV_32FC2);
    for (int y = 0; y < prev.rows; ++y) {
        const float* pu = u.ptr<float>(y);
        const float* pv = v.ptr<float>(y);
        Vec2f* pf = flow.ptr<Vec2f>(y);
        for (int x = 0; x < prev.cols; ++x)
            pf[x] = Vec2f(pu[x], pv[x]);
    }
    return flow;
}

Point2f estimateGlobalTranslation(const Mat& flow) {
    Mat ch[2]; split(flow, ch);
    Scalar mu = mean(ch[0]);
    Scalar mv = mean(ch[1]);
    return Point2f((float)mu[0], (float)mv[0]);
}

Point2f estimateGlobalTranslationRANSAC(const Mat& prevGray, const Mat& gray) {
    std::vector<Point2f> ptsPrev;
    goodFeaturesToTrack(prevGray, ptsPrev, 200, 0.01, 10);
    if (ptsPrev.empty()) return Point2f(0,0);
    std::vector<Point2f> ptsNext;
    std::vector<uchar> status;
    std::vector<float> err;
    calcOpticalFlowPyrLK(prevGray, gray, ptsPrev, ptsNext, status, err);
    std::vector<Point2f> src, dst;
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
            src.push_back(ptsPrev[i]);
            dst.push_back(ptsNext[i]);
        }
    }
    if (src.size() < 6) return Point2f(0,0);
    std::vector<uchar> inliers;
    Mat affine = estimateAffinePartial2D(src, dst, inliers, RANSAC, 5.0);
    if (affine.empty()) return Point2f(0,0);
    return Point2f((float)affine.at<double>(0,2), (float)affine.at<double>(1,2));
}

Mat compensateTranslation(const Mat& frame, const Point2f& t) {
    Mat M = (Mat_<double>(2,3) << 1,0,-t.x, 0,1,-t.y);
    Mat out;
    warpAffine(frame, out, M, frame.size(), INTER_LINEAR, BORDER_REFLECT);
    return out;
}

Mat detectMovingMask(const Mat& flow, float thresh) {
    Mat mag, ang;
    Mat ch[2]; split(flow, ch);
    cartToPolar(ch[0], ch[1], mag, ang, false);
    Mat mask;
    threshold(mag, mask, thresh, 255, THRESH_BINARY);
    mask.convertTo(mask, CV_8U);
    Mat ker = getStructuringElement(MORPH_ELLIPSE, Size(7,7));
    morphologyEx(mask, mask, MORPH_CLOSE, ker);
    morphologyEx(mask, mask, MORPH_OPEN,  ker);
    return mask;
}

Mat removeMovingObjects(const Mat& frame, const Mat& mask, int inpaintRadius) {
    Mat inpainted;
    inpaint(frame, mask, inpainted, inpaintRadius, INPAINT_TELEA);
    return inpainted;
}

Mat dimMaskRegion(const Mat& frame, const Mat& mask, float alpha) {
    CV_Assert(alpha >= 0.0f && alpha <= 1.0f);
    std::vector<std::vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    Mat contourMask = Mat::zeros(mask.size(), CV_8U);
    drawContours(contourMask, contours, -1, Scalar(255), FILLED);
    Mat out = frame.clone();
    out.forEach<Vec3b>([&](Vec3b &pixel, const int pos[]) {
        int y = pos[0], x = pos[1];
        if (contourMask.at<uchar>(y,x)) {
            pixel[0] = saturate_cast<uchar>(pixel[0] * alpha);
            pixel[1] = saturate_cast<uchar>(pixel[1] * alpha);
            pixel[2] = saturate_cast<uchar>(pixel[2] * alpha);
        }
    });
    return out;
}
