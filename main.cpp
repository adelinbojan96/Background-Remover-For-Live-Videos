#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>
#include <array>
#include <stdexcept>

using namespace std;
using namespace cv;

const string youtubeLink = "https://www.youtube.com/watch?v=iJ0GXMSDPsM&ab_channel=RailCamNetherlands";
const double alpha = 0.008;

string getYoutubeStreamURL(const string& videoLink) {
    array<char,4096> buffer{};
    string result;
    const string cmd = "yt-dlp -g \"" + videoLink + "\"";
    const shared_ptr<FILE> pipe(_popen(cmd.c_str(), "r"), _pclose);
    if (!pipe) throw runtime_error("yt-dlp popen() failed");
    while (fgets(buffer.data(), buffer.size(), pipe.get()))
        result += buffer.data();
    if (!result.empty() && result.back() == '\n')
        result.pop_back();
    return result;
}

void processStream(const string& streamURL) {
    VideoCapture cap(streamURL, CAP_FFMPEG);
    if (!cap.isOpened()) {
        cerr << "error -> Cannot open stream\n";
        return;
    }

    const Size targetSize(1280,720);
    Mat raw, frame, gray, bgModelF, bgModel, diff, fgMask, cleanMask;

    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3,3));
    bool initialized = false;

    while (true) {
        if (!cap.read(raw) || raw.empty()) break;
        resize(raw, frame, targetSize);
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        if (!initialized) {
            gray.convertTo(bgModelF, CV_32F);
            initialized = true;
        } else {
            Mat grayF;
            gray.convertTo(grayF, CV_32F);
            accumulateWeighted(grayF, bgModelF, alpha);
        }

        bgModelF.convertTo(bgModel, CV_8U);
        absdiff(gray, bgModel, diff);
        threshold(diff, fgMask, 25, 255, THRESH_BINARY);
        morphologyEx(fgMask, cleanMask, MORPH_OPEN, kernel);

        Mat greenBg(frame.size(), frame.type(), Scalar(0,255,0));
        Mat result = frame.clone();
        greenBg.copyTo(result, cleanMask == 0);

        imshow("Live Stream", frame);
        imshow("Background Result", result);

        if (waitKey(1) == 27) break;
    }

    cap.release();
    destroyAllWindows();
}

int main() {
    try {
        cout << "fetching stream URL...\n";
        const string url = getYoutubeStreamURL(youtubeLink);
        cout << "stream URL: " << url << "\n";
        processStream(url);
    } catch (const exception& e) {
        cerr << e.what() << "\n";
        return -1;
    }
    return 0;
}
