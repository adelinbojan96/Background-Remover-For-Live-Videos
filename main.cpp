#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>
#include <array>
#include <stdexcept>

using namespace std;
using namespace cv;

string getYoutubeStreamURL(const string& videoLink) {
    array<char, 1024> buffer;
    string result;
    string cmd = "yt-dlp -g \"" + videoLink + "\"";
    shared_ptr<FILE> pipe(_popen(cmd.c_str(), "r"), _pclose);
    if (!pipe) throw runtime_error("yt-dlp popen() failed");
    while (fgets(buffer.data(), buffer.size(), pipe.get())) {
        result += buffer.data();
    }
    if (!result.empty() && result.back() == '\n')
        result.pop_back();
    return result;
}

Ptr<BackgroundSubtractor> initBackgroundSubtractor(double history = 500,
                                                   double varThreshold = 16,
                                                   bool detectShadows = true) {
    return createBackgroundSubtractorMOG2(history, varThreshold, detectShadows);
}

void processStream(const string& streamURL, const Mat& newBg) {
    VideoCapture cap(streamURL);
    if (!cap.isOpened()) {
        cerr << "ERROR: Cannot open stream\n";
        return;
    }

    Ptr<BackgroundSubtractor> backSub = initBackgroundSubtractor();

    Size sz((int)cap.get(CAP_PROP_FRAME_WIDTH),
            (int)cap.get(CAP_PROP_FRAME_HEIGHT));
    Mat frame(sz, CV_8UC3),
        fgMask(sz, CV_8UC1),
        fgMaskClean(sz, CV_8UC1),
        composite(sz, CV_8UC3),
        background;
    resize(newBg, background, sz);

    // Pre-built morphology kernels
    Mat kernelOpen   = getStructuringElement(MORPH_ELLIPSE, Size(3,3));
    Mat kernelDilate = getStructuringElement(MORPH_ELLIPSE, Size(5,5));

    while (true) {
        if (!cap.read(frame) || frame.empty()) break;

        backSub->apply(frame, fgMask);
        morphologyEx(fgMask,     fgMaskClean, MORPH_OPEN,   kernelOpen);
        morphologyEx(fgMaskClean,fgMaskClean, MORPH_DILATE, kernelDilate);

        background.copyTo(composite);
        frame.copyTo(composite, fgMaskClean);

        imshow("Composite", composite);
        if (waitKey(1) == 27) break;  // ESC
    }
    cap.release();
    destroyAllWindows();
}

int main() {
    try {
        string youtubeLink = "https://www.youtube.com/watch?v=rnXIjl_Rzy4&ab_channel=EarthCam";
        cout << "Fetching stream URL...\n";
        string streamURL = getYoutubeStreamURL(youtubeLink);
        cout << "Stream URL: " << streamURL << "\n";

        Mat newBg = imread("background.jpg");
        if (newBg.empty()) {
            cerr << "ERROR: Cannot load background.jpg\n";
            return -1;
        }

        processStream(streamURL, newBg);
    }
    catch (const exception& e) {
        cerr << "Exception: " << e.what() << "\n";
        return -1;
    }
    return 0;
}
