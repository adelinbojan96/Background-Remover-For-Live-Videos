#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <stdexcept>

using namespace std;
using namespace cv;

const double ALPHA = 0.005;

Mat erode(const Mat &src, const Mat &kernel) {
    if (src.type() != CV_8UC1 || kernel.type() != CV_8UC1)
        throw runtime_error("erode: only CV_8UC1 supported");

    Mat dst(src.size(), src.type(), Scalar(255));
    int kRows = kernel.rows, kCols = kernel.cols;
    int kCenterY = kRows/2, kCenterX = kCols/2;

    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            uchar minVal = 255;
            for (int m = 0; m < kRows; ++m) {
                for (int n = 0; n < kCols; ++n) {
                    if (kernel.at<uchar>(m,n)) {
                        int y = i + m - kCenterY;
                        int x = j + n - kCenterX;
                        if (y>=0 && y<src.rows && x>=0 && x<src.cols) {
                            minVal = min(minVal, src.at<uchar>(y,x));
                        }
                    }
                }
            }
            dst.at<uchar>(i,j) = minVal;
        }
    }
    return dst;
}

Mat dilate(const Mat &src, const Mat &kernel) {
    if (src.type() != CV_8UC1 || kernel.type() != CV_8UC1)
        throw runtime_error("dilate: only CV_8UC1 supported");

    Mat dst(src.size(), src.type(), Scalar(0));
    int kRows = kernel.rows, kCols = kernel.cols;
    int kCenterY = kRows/2, kCenterX = kCols/2;

    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            uchar maxVal = 0;
            for (int m = 0; m < kRows; ++m) {
                for (int n = 0; n < kCols; ++n) {
                    if (kernel.at<uchar>(m,n)) {
                        int y = i + m - kCenterY;
                        int x = j + n - kCenterX;
                        if (y>=0 && y<src.rows && x>=0 && x<src.cols) {
                            maxVal = max(maxVal, src.at<uchar>(y,x));
                        }
                    }
                }
            }
            dst.at<uchar>(i,j) = maxVal;
        }
    }
    return dst;
}

Mat captureBackground(const string& videoPath,
                      int history = 2000,
                      double learningRate = 0.05) {
    VideoCapture cap(videoPath);
    if (!cap.isOpened())
        throw runtime_error("captureBackground: cannot open file " + videoPath);

    Mat frame;
    if (!cap.read(frame) || frame.empty())
        throw runtime_error("captureBackground: cannot read first frame");

    Size targetSize(1280, 720);
    Mat resized;
    resize(frame, resized, targetSize);

    Mat backgroundF(targetSize, CV_32FC3);
    for (int y = 0; y < targetSize.height; ++y) {
        for (int x = 0; x < targetSize.width; ++x) {
            Vec3b p = resized.at<Vec3b>(y,x);
            Vec3f &b = backgroundF.at<Vec3f>(y,x);
            b[0]=p[0]; b[1]=p[1]; b[2]=p[2];
        }
    }

    for (int i = 1; i < history; ++i) {
        if (!cap.read(frame) || frame.empty()) break;
        resize(frame, resized, targetSize);
        for (int y = 0; y < targetSize.height; ++y) {
            Vec3b const* srcRow = resized.ptr<Vec3b>(y);
            Vec3f      * bgRow  = backgroundF.ptr<Vec3f>(y);
            for (int x = 0; x < targetSize.width; ++x) {
                for (int c = 0; c < 3; ++c) {
                    bgRow[x][c] = float((1.0 - learningRate) * bgRow[x][c]
                                     + learningRate * srcRow[x][c]);
                }
            }
        }
    }

    Mat background8U(targetSize, CV_8UC3);
    for (int y = 0; y < targetSize.height; ++y) {
        for (int x = 0; x < targetSize.width; ++x) {
            Vec3f f = backgroundF.at<Vec3f>(y,x);
            background8U.at<Vec3b>(y,x) = Vec3b(
                uchar(round(f[0])), uchar(round(f[1])), uchar(round(f[2]))
            );
        }
    }

    cap.release();
    return background8U;
}

Mat ellipseKernel(int ksize = 3) {
    if (ksize < 1 || ksize % 2 == 0)
        throw runtime_error("ellipseKernel: ksize must be odd > 0");

    Mat kernel(ksize, ksize, CV_8UC1, Scalar(0));
    double r = (ksize - 1) / 2.0;
    for (int i = 0; i < ksize; ++i) {
        for (int j = 0; j < ksize; ++j) {
            double dx = (j - r) / r, dy = (i - r) / r;
            if (dx*dx + dy*dy <= 1.0)
                kernel.at<uchar>(i,j) = 1;
        }
    }
    return kernel;
}

void processStream(const string& videoPath, const Mat& staticBg) {
    VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        cerr << "processStream: cannot open file " << videoPath << "\n";
        return;
    }

    const Size targetSize(1280, 720);
    Mat raw, frame, gray, bgModelF, bgModel, diff, fgMask, cleanMask;
    Mat kernel = ellipseKernel();
    bool initialized = false;

    Mat bgResized;
    if (!staticBg.empty()) {
        resize(staticBg, bgResized, targetSize);
    }

    while (true) {
        if (!cap.read(raw) || raw.empty()) break;

        resize(raw, frame, targetSize);
        gray.create(frame.size(), CV_8UC1);
        for (int y = 0; y < frame.rows; ++y) {
            for (int x = 0; x < frame.cols; ++x) {
                Vec3b bgr = frame.at<Vec3b>(y,x);
                gray.at<uchar>(y,x) = uchar(0.114*bgr[0] + 0.587*bgr[1] + 0.299*bgr[2]);
            }
        }

        if (!initialized) {
            gray.convertTo(bgModelF, CV_32F);
            initialized = true;
        } else {
            Mat grayF; gray.convertTo(grayF, CV_32F);
            for (int y = 0; y < grayF.rows; ++y) {
                float* bRow = bgModelF.ptr<float>(y);
                float const* gRow = grayF.ptr<float>(y);
                for (int x = 0; x < grayF.cols; ++x)
                    bRow[x] = (1.0f - ALPHA)*bRow[x] + ALPHA*gRow[x];
            }
        }

        bgModelF.convertTo(bgModel, CV_8U);
        diff.create(gray.size(), CV_8UC1);
        for (int y = 0; y < gray.rows; ++y) {
            for (int x = 0; x < gray.cols; ++x) {
                diff.at<uchar>(y,x) = uchar(abs(int(gray.at<uchar>(y,x))
                                               - int(bgModel.at<uchar>(y,x))));
            }
        }

        fgMask.create(diff.size(), CV_8UC1);
        for (int y = 0; y < diff.rows; ++y) {
            for (int x = 0; x < diff.cols; ++x) {
                fgMask.at<uchar>(y,x) = diff.at<uchar>(y,x) > 25 ? 255 : 0;
            }
        }

        Mat erodedMask = erode(fgMask, kernel);
        cleanMask = dilate(erodedMask, kernel);

        Mat greenBg(frame.size(), frame.type(), Scalar(0,255,0));
        Mat greenResult = frame.clone();
        for (int y = 0; y < frame.rows; ++y) {
            for (int x = 0; x < frame.cols; ++x) {
                if (cleanMask.at<uchar>(y,x) == 0)
                    greenResult.at<Vec3b>(y,x) = greenBg.at<Vec3b>(y,x);
            }
        }
        imshow("Green-Screen Result", greenResult);

        if (!bgResized.empty()) {
            Mat staticResult = frame.clone();
            for (int y = 0; y < frame.rows; ++y) {
                for (int x = 0; x < frame.cols; ++x) {
                    if (cleanMask.at<uchar>(y,x))
                        staticResult.at<Vec3b>(y,x) = bgResized.at<Vec3b>(y,x);
                }
            }
            imshow("Static Background Removed", staticResult);
        }

        imshow("Live Video", frame);
        if (waitKey(10) == 27) break;
    }

    cap.release();
    destroyAllWindows();
}