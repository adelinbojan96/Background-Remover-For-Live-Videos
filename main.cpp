#include <opencv2/opencv.hpp>
#include <opencv2/video.hpp>
#include <iostream>
#include <memory>
#include <array>
#include <stdexcept>

using namespace std;
using namespace cv;

const double ALPHA = 0.05;

string getYoutubeStreamURL(const string& videoLink) {
    array<char,4096> buffer{};
    string result;
    string cmd = "yt-dlp -g \"" + videoLink + "\"";
    shared_ptr<FILE> pipe(_popen(cmd.c_str(), "r"), _pclose);
    if (!pipe)
        throw runtime_error("yt-dlp popen() failed");

    while (fgets(buffer.data(), buffer.size(), pipe.get()))
        result += buffer.data();

    if (!result.empty() && result.back() == '\n')
        result.pop_back();

    return result;
}

Mat captureBackground(const string& streamURL, int history = 200, double learningRate = 0.03) {
    VideoCapture cap(streamURL, CAP_FFMPEG);
    if (!cap.isOpened())
        throw runtime_error("captureBackground: cannot open stream");

    Ptr<BackgroundSubtractorMOG2> pBackSub = createBackgroundSubtractorMOG2(history, 16, false);
    Mat frame, fgMask;

    for (int i = 0; i < history; i++) {
        if (!cap.read(frame) || frame.empty())
            break;
        resize(frame, frame, Size(1280, 720));
        pBackSub->apply(frame, fgMask, learningRate);
    }

    Mat background;
    pBackSub->getBackgroundImage(background);
    cap.release();
    return background;
}

void processStream(const string& streamURL, const Mat& staticBg) {
    VideoCapture cap(streamURL, CAP_FFMPEG);
    if (!cap.isOpened()) {
        cerr << "error -> Cannot open stream\n";
        return;
    }

    const Size targetSize(1280, 720);
    Mat raw, frame;
    Mat gray, bgModelF, bgModel, diff, fgMask, cleanMask;
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    bool initialized = false;

    Mat bgResized;
    if (!staticBg.empty())
        resize(staticBg, bgResized, targetSize);

    while (true) {
        if (!cap.read(raw) || raw.empty())
            break;
        resize(raw, frame, targetSize);

        // grayscale for green screen
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        if (!initialized) {
            gray.convertTo(bgModelF, CV_32F);
            initialized = true;
        } else {
            Mat grayF; gray.convertTo(grayF, CV_32F);
            accumulateWeighted(grayF, bgModelF, ALPHA);
        }
        bgModelF.convertTo(bgModel, CV_8U);
        absdiff(gray, bgModel, diff);
        threshold(diff, fgMask, 25, 255, THRESH_BINARY);
        morphologyEx(fgMask, cleanMask, MORPH_OPEN, kernel);

        // green-screen replacement
        Mat greenBg(frame.size(), frame.type(), Scalar(0, 255, 0));
        Mat greenResult = frame.clone();
        greenBg.copyTo(greenResult, cleanMask == 0);

        imshow("Green-Screen Result", greenResult);

        if (!staticBg.empty()) {
            //static background removal
            Mat staticResult = frame.clone();
            bgResized.copyTo(staticResult, cleanMask);
            imshow("Static Background Removed", staticResult);
        }

        imshow("Live Stream", frame);

        if (waitKey(10) == 27) break;
    }

    cap.release();
    destroyAllWindows();
}

int main() {
    try {
        vector<pair<string, string>> streams = {
            { "RailCam Netherlands", "https://www.youtube.com/watch?v=iJ0GXMSDPsM&ab_channel=RailCamNetherlands" },
            { "EarthCam – Times Square", "https://www.youtube.com/watch?v=j9Sa4uBGGQ0&ab_channel=EarthCam" },
            { "Amsterdam Schiphol Airport", "https://www.youtube.com/watch?v=BTbdzpWBkg0&ab_channel=AMSLIVE" }
        };
        cout << "Please select a live stream:\n";
        for (size_t i = 0; i < streams.size(); ++i)
            cout << i+1 << ". " << streams[i].first << "\n";

        int choice = 0;
        while (choice < 1 || choice > streams.size()) {
            cout << "Enter choice [1-" << streams.size() << "]: "; cin >> choice;
        }

        const string url = getYoutubeStreamURL(streams[choice-1].second);
        cout << "Stream URL: " << url << "\n";

        Mat staticBg;
        if (choice == 1 || choice == 2) {
            cout << "Capturing static background (this may take a few seconds)...\n";
            staticBg = captureBackground(url, 200, 0.01);
            imshow("Captured Static Background", staticBg);
            waitKey(1000);
        }

        processStream(url, staticBg);
    } catch (const exception& e) {
        cerr << e.what() << "\n";
        return -1;
    }
    return 0;
}
