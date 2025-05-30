#include "dynamic_camera.h"
#include <opencv2/photo.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <random>
#include <cmath>

using namespace cv;
using namespace std;

void computeGradients(const Mat& prev, const Mat& next, Mat& Ix, Mat& Iy, Mat& It) {
    int rows = prev.rows;
    int cols = prev.cols;

    Ix = Mat::zeros(rows, cols, CV_32F);
    Iy = Mat::zeros(rows, cols, CV_32F);
    It = Mat::zeros(rows, cols, CV_32F);

    const float sobelX[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    const float sobelY[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    for (int y = 1; y < rows - 1; ++y) {
        for (int x = 1; x < cols - 1; ++x) {
            float gx = 0.0f, gy = 0.0f;

            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {

                    float p = prev.at<uchar>(y + ky, x + kx);
                    gx += sobelX[ky + 1][kx + 1] * p;
                    gy += sobelY[ky + 1][kx + 1] * p;
                }
            }

            Ix.at<float>(y, x) = gx;
            Iy.at<float>(y, x) = gy;
            It.at<float>(y, x) = (float)(next.at<uchar>(y, x) - prev.at<uchar>(y, x));
        }
    }
}

void boxBlur3x3(const Mat& src, Mat& dst) {
    int rows = src.rows;
    int cols = src.cols;
    dst = Mat::zeros(rows, cols, CV_32F);

    for (int y = 1; y < rows - 1; ++y) {
        for (int x = 1; x < cols - 1; ++x) {
            float sum = 0.0f;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    sum += src.at<float>(y + dy, x + dx);
                }
            }
            dst.at<float>(y, x) = sum / 9.0f;
        }
    }
}

void elementwiseProduct(const Mat& a, const Mat& b, Mat& out) {
    int rows = a.rows, cols = a.cols;
    out = Mat::zeros(rows, cols, CV_32F);
    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x)
            out.at<float>(y, x) = a.at<float>(y, x) * b.at<float>(y, x);
}

void elementwiseAdd(const Mat& a, const Mat& b, Mat& out) {
    int rows = a.rows, cols = a.cols;
    out = Mat::zeros(rows, cols, CV_32F);
    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x)
            out.at<float>(y, x) = a.at<float>(y, x) + b.at<float>(y, x);
}

void elementwiseSub(const Mat& a, const Mat& b, Mat& out) {
    int rows = a.rows, cols = a.cols;
    out = Mat::zeros(rows, cols, CV_32F);
    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x)
            out.at<float>(y, x) = a.at<float>(y, x) - b.at<float>(y, x);
}

void elementwiseDiv(const Mat& a, const Mat& b, Mat& out) {
    int rows = a.rows, cols = a.cols;
    out = Mat::zeros(rows, cols, CV_32F);
    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x)
            out.at<float>(y, x) = a.at<float>(y, x) / b.at<float>(y, x);
}

void addScalar(const Mat& a, float s, Mat& out) {
    int rows = a.rows, cols = a.cols;
    out = Mat::zeros(rows, cols, CV_32F);
    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x)
            out.at<float>(y, x) = a.at<float>(y, x) + s;
}

void mulScalar(const Mat& a, float s, Mat& out) {
    int rows = a.rows, cols = a.cols;
    out = Mat::zeros(rows, cols, CV_32F);
    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x)
            out.at<float>(y, x) = a.at<float>(y, x) * s;
}

Mat hornSchunck(const Mat& prev, const Mat& next, float alpha, int numIter) {
    Mat Ix, Iy, It;
    computeGradients(prev, next, Ix, Iy, It);

    int rows = prev.rows, cols = prev.cols;
    Mat u = Mat::zeros(prev.size(), CV_32F);
    Mat v = Mat::zeros(prev.size(), CV_32F);
    Mat uAvg, vAvg;

    Mat Ix2, Iy2, denom1, denom, Ixu, Iyv, term1, term, Ixt, Ixt_div, u_new, Iyt, Iyt_div, v_new;

    for (int k = 0; k < numIter; ++k) {
        boxBlur3x3(u, uAvg);
        boxBlur3x3(v, vAvg);

        // denom = alpha*alpha + Ix * Ix + Iy * Iy
        elementwiseProduct(Ix, Ix, Ix2);
        elementwiseProduct(Iy, Iy, Iy2);
        addScalar(Ix2, alpha * alpha, denom1);
        elementwiseAdd(denom1, Iy2, denom);

        // term = Ix * (uAvg) + Iy * (vAvg) + It;
        elementwiseProduct(Ix, uAvg, Ixu);
        elementwiseProduct(Iy, vAvg, Iyv);
        elementwiseAdd(Ixu, Iyv, term1);
        elementwiseAdd(term1, It, term);

        // u = uAvg - Ix * (term) / denom;
        elementwiseProduct(Ix, term, Ixt);
        elementwiseDiv(Ixt, denom, Ixt_div);
        elementwiseSub(uAvg, Ixt_div, u_new);
        u = u_new;

        // v = vAvg - Iy * (term) / denom;
        elementwiseProduct(Iy, term, Iyt);
        elementwiseDiv(Iyt, denom, Iyt_div);
        elementwiseSub(vAvg, Iyt_div, v_new);
        v = v_new;
    }

    Mat flow(prev.size(), CV_32FC2);
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            flow.at<Vec2f>(y, x)[0] = u.at<float>(y, x);
            flow.at<Vec2f>(y, x)[1] = v.at<float>(y, x);
        }
    }
    return flow;
}

Point2f estimateGlobalTranslation(const Mat& flow) {
    double sumX = 0.0, sumY = 0.0;
    int count = 0;
    for (int y = 0; y < flow.rows; ++y) {
        for (int x = 0; x < flow.cols; ++x) {
            const auto& f = flow.at<Vec2f>(y, x);
            sumX += f[0];
            sumY += f[1];
            ++count;
        }
    }
    const float meanX = count > 0 ? (float)(sumX / count) : 0.f;
    const float meanY = count > 0 ? (float)(sumY / count) : 0.f;
    return {meanX, meanY};
}



Point2f estimateGlobalTranslationRANSAC(const Mat& prevGray, const Mat& gray) {
    vector<Point2f> ptsPrev;
    int step = 20;
    for (int y = step; y < prevGray.rows - step; y += step)
        for (int x = step; x < prevGray.cols - step; x += step)
            ptsPrev.emplace_back((float)x, (float)y);

    vector<Point2f> ptsNext(ptsPrev.size());
    vector<uchar> status(ptsPrev.size(), 0);
    int win = 5;
    for (size_t i = 0; i < ptsPrev.size(); ++i) {
        int x0 = cvRound(ptsPrev[i].x), y0 = cvRound(ptsPrev[i].y);
        int bestDx = 0, bestDy = 0, minSAD = INT_MAX;

        for (int dy = -win; dy <= win; ++dy) {
            for (int dx = -win; dx <= win; ++dx) {
                int x1 = x0 + dx, y1 = y0 + dy;
                if (x1 < win || x1 >= gray.cols - win || y1 < win || y1 >= gray.rows - win)
                    continue;

                int sad = 0;
                for (int wy = -win; wy <= win; ++wy)
                    for (int wx = -win; wx <= win; ++wx) {
                        int val0 = prevGray.at<uchar>(y0 + wy, x0 + wx);
                        int val1 = gray.at<uchar>(y1 + wy, x1 + wx);
                        sad += std::abs(val0 - val1);
                    }

                if (sad < minSAD) {
                    minSAD = sad;
                    bestDx = dx;
                    bestDy = dy;
                }
            }
        }
        if (minSAD < 5000) {
            ptsNext[i] = Point2f(x0 + bestDx, y0 + bestDy);
            status[i] = 1;
        }
    }

    vector<Point2f> src, dst;
    for (size_t i = 0; i < status.size(); ++i)
        if (status[i]) {
            src.push_back(ptsPrev[i]);
            dst.push_back(ptsNext[i]);
        }
    if (src.size() < 6)
        return {0,0};

    int bestInliers = 0;
    float bestTx = 0, bestTy = 0;
    int N = 100;
    float threshold = 3.0f;

    //random number generator
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, src.size() - 1);

    for (int iter = 0; iter < N; ++iter) {
        int idx = dis(gen);
        float tx = dst[idx].x - src[idx].x;
        float ty = dst[idx].y - src[idx].y;
        int inliers = 0;
        for (size_t i = 0; i < src.size(); ++i) {
            float dx = dst[i].x - src[i].x;
            float dy = dst[i].y - src[i].y;
            float dist = std::sqrt((dx - tx)*(dx - tx) + (dy - ty)*(dy - ty));
            if (dist < threshold)
                ++inliers;
        }
        if (inliers > bestInliers) {
            bestInliers = inliers;
            bestTx = tx;
            bestTy = ty;
        }
    }

    float sumTx = 0, sumTy = 0;
    int n = 0;
    for (size_t i = 0; i < src.size(); ++i) {
        float dx = dst[i].x - src[i].x;
        float dy = dst[i].y - src[i].y;
        float dist = std::sqrt((dx - bestTx)*(dx - bestTx) + (dy - bestTy)*(dy - bestTy));
        if (dist < threshold) {
            sumTx += dx;
            sumTy += dy;
            ++n;
        }
    }
    if (n == 0) return {0,0};
    return {sumTx/n, sumTy/n};
}


int reflect(int p, int limit) {
    if (p < 0)
        return -p - 1;
    if (p >= limit)
        return 2 * limit - p - 1;
    return p;
}

float interp1ch(const Mat& img, float y, float x) {
    //bilinear interpolation for single channel images
    int rows = img.rows, cols = img.cols;
    int x0 = floor(x), y0 = floor(y);
    int x1 = x0 + 1, y1 = y0 + 1;

    float a = x - x0;
    float b = y - y0;

    x0 = reflect(x0, cols);
    x1 = reflect(x1, cols);
    y0 = reflect(y0, rows);
    y1 = reflect(y1, rows);

    float v00 = img.at<uchar>(y0, x0);
    float v01 = img.at<uchar>(y0, x1);
    float v10 = img.at<uchar>(y1, x0);
    float v11 = img.at<uchar>(y1, x1);

    float val = (1 - a) * (1 - b) * v00 + a * (1 - b) * v01 +
                (1 - a) * b * v10 + a * b * v11;
    return max(0.f, min(255.f, val));
}

Vec3b interp3ch(const Mat& img, float y, float x) {
    //bilinear interpolation for 3-channel images
    int rows = img.rows, cols = img.cols;
    int x0 = floor(x), y0 = floor(y);
    int x1 = x0 + 1, y1 = y0 + 1;

    float a = x - x0;
    float b = y - y0;

    x0 = reflect(x0, cols);
    x1 = reflect(x1, cols);
    y0 = reflect(y0, rows);
    y1 = reflect(y1, rows);

    Vec3b v00 = img.at<Vec3b>(y0, x0);
    Vec3b v01 = img.at<Vec3b>(y0, x1);
    Vec3b v10 = img.at<Vec3b>(y1, x0);
    Vec3b v11 = img.at<Vec3b>(y1, x1);

    Vec3b val;
    for (int c = 0; c < 3; ++c) {
        float interp =
            (1 - a) * (1 - b) * v00[c] + a * (1 - b) * v01[c] +
            (1 - a) * b * v10[c] + a * b * v11[c];
        val[c] = uchar(interp);
    }
    return val;
}

Mat compensateTranslation(const Mat& frame, const Point2f& t) {
    Mat out = Mat::zeros(frame.size(), frame.type());
    const int rows = frame.rows;
    const int cols = frame.cols;
    const bool isColor = frame.channels() == 3;

    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            const float srcX = x + t.x;
            const float srcY = y + t.y;
            if (isColor) {
                out.at<Vec3b>(y, x) = interp3ch(frame, srcY, srcX);
            } else {
                out.at<uchar>(y, x) = (uchar)(interp1ch(frame, srcY, srcX) + 0.5f);
            }
        }
    }
    return out;
}

void dilate(const Mat& src, Mat& dst, const Mat& kernel) {
    dst = Mat::zeros(src.size(), src.type());
    const int kRows = kernel.rows;
    const int kCols = kernel.cols;
    const int kr = kRows / 2;
    const int kc = kCols / 2;
    for (int y = kr; y < src.rows - kr; ++y) {
        for (int x = kc; x < src.cols - kc; ++x) {
            uchar maxVal = 0;
            for (int i = -kr; i <= kr; ++i) {
                for (int j = -kc; j <= kc; ++j) {
                    if (kernel.at<uchar>(i + kr, j + kc)) {
                        maxVal = max(maxVal, src.at<uchar>(y + i, x + j));
                    }
                }
            }
            dst.at<uchar>(y, x) = maxVal;
        }
    }
}

void erode(const Mat& src, Mat& dst, const Mat& kernel) {
    dst = Mat::zeros(src.size(), src.type());
    const int kRows = kernel.rows;
    const int kCols = kernel.cols;
    const int kr = kRows / 2;
    const int kc = kCols / 2;
    for (int y = kr; y < src.rows - kr; ++y) {
        for (int x = kc; x < src.cols - kc; ++x) {
            uchar minVal = 255;
            for (int i = -kr; i <= kr; ++i) {
                for (int j = -kc; j <= kc; ++j) {
                    if (kernel.at<uchar>(i + kr, j + kc)) {
                        minVal = min(minVal, src.at<uchar>(y + i, x + j));
                    }
                }
            }
            dst.at<uchar>(y, x) = minVal;
        }
    }
}

Mat detectMovingMask(const Mat& flow, float thresh) {
    Mat mask = Mat::zeros(flow.rows, flow.cols, CV_8U);

    for (int y = 0; y < flow.rows; ++y) {
        for (int x = 0; x < flow.cols; ++x) {
            const Vec2f& f = flow.at<Vec2f>(y, x);
            float mag = sqrt(f[0] * f[0] + f[1] * f[1]);
            if (mag > thresh) {
                mask.at<uchar>(y, x) = 255;
            }
        }
    }

    uchar ker[7][7] = {
        {0,0,1,1,1,0,0},
        {0,1,1,1,1,1,0},
        {1,1,1,1,1,1,1},
        {1,1,1,1,1,1,1},
        {1,1,1,1,1,1,1},
        {0,1,1,1,1,1,0},
        {0,0,1,1,1,0,0}
    };

    const Mat kerBin(7, 7, CV_8U, ker);
    Mat temp;

    dilate(mask, temp, kerBin);
    erode(temp, mask, kerBin);

    erode(mask, temp, kerBin);
    dilate(temp, mask, kerBin);

    return mask;
}

Mat removeMovingObjects(const Mat& frame, const Mat& mask, int inpaint_radius) {
    Mat inpainted;
    inpaint(frame, mask, inpainted, inpaint_radius, INPAINT_TELEA);
    return inpainted;
}

Mat dimMaskRegionLowLevel(const Mat& frame, const Mat& mask, float alpha) {
    CV_Assert(alpha >= 0.0f && alpha <= 1.0f);
    Mat out = frame.clone();
    for (int y = 0; y < frame.rows; ++y) {
        for (int x = 0; x < frame.cols; ++x) {
            if (mask.at<uchar>(y, x)) {
                auto& pix = out.at<Vec3b>(y, x);
                pix[0] = (uchar)(pix[0] * alpha);
                pix[1] = (uchar)(pix[1] * alpha);
                pix[2] = (uchar)(pix[2] * alpha);
            }
        }
    }
    return out;
}

vector<vector<float>> createGaussianKernel(int ksize, float sigma) {
    int half = ksize / 2;

    vector kernel(ksize, vector<float>(ksize));
    float sum = 0.0f;
    for (int y = -half; y <= half; ++y) {
        for (int x = -half; x <= half; ++x) {
            const float val = std::exp(-(x*x + y*y) / (2 * sigma * sigma));
            kernel[y + half][x + half] = val;
            sum += val;
        }
    }
    // normalisation
    for (int y = 0; y < ksize; ++y)
        for (int x = 0; x < ksize; ++x)
            kernel[y][x] /= sum;
    return kernel;
}

Mat gaussianBlurGray(const Mat& src, int ksize, float sigma) {
    int half = ksize / 2;
    auto kernel = createGaussianKernel(ksize, sigma);
    Mat dst = src.clone();

    for (int y = half; y < src.rows - half; ++y) {
        for (int x = half; x < src.cols - half; ++x) {
            float sum = 0;
            for (int ky = -half; ky <= half; ++ky)
                for (int kx = -half; kx <= half; ++kx)
                    sum += src.at<uchar>(y + ky, x + kx) * kernel[ky + half][kx + half];
            dst.at<uchar>(y, x) = (uchar)sum;
        }
    }
    return dst;
}

Mat gaussianBlurColor(const Mat& src, int ksize, float sigma) {
    const int half = ksize / 2;
    const auto kernel = createGaussianKernel(ksize, sigma);
    Mat dst = src.clone();

    for (int y = half; y < src.rows - half; ++y) {
        for (int x = half; x < src.cols - half; ++x) {
            float sum[3] = {0,0,0};
            for (int ky = -half; ky <= half; ++ky) {
                for (int kx = -half; kx <= half; ++kx) {
                    Vec3b pix = src.at<Vec3b>(y + ky, x + kx);
                    float w = kernel[ky + half][kx + half];
                    sum[0] += pix[0] * w;
                    sum[1] += pix[1] * w;
                    sum[2] += pix[2] * w;
                }
            }
            dst.at<Vec3b>(y, x) = Vec3b(
                (uchar)(sum[0]),
                (uchar)(sum[1]),
                (uchar)(sum[2])
            );
        }
    }
    return dst;
}

Mat gaussianBlur(const Mat& src, int ksize, float sigma) {
    if (src.type() == CV_8UC1)
        return gaussianBlurGray(src, ksize, sigma);
    if (src.type() == CV_8UC3)
        return gaussianBlurColor(src, ksize, sigma);
    throw runtime_error("Unsupported image type for Gaussian blur");
}

Mat ellipseKernel_2(int ksize = 3) {
    if (ksize < 1 || ksize % 2 == 0)
        throw runtime_error("ellipseKernel: ksize must be odd and > 0");

    Mat kernel(ksize, ksize, CV_8UC1, Scalar(0));
    double r = (ksize - 1) / 2.0;
    for (int i = 0; i < ksize; ++i) {
        for (int j = 0; j < ksize; ++j) {
            double dx = (j - r) / r, dy = (i - r) / r;
            if (dx*dx + dy*dy <= 1.0)
                kernel.at<uchar>(i, j) = 1;
        }
    }
    return kernel;
}
