#include "static_camera.h"
#include "dynamic_camera.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
using namespace std;
using namespace cv;
using namespace dnn;

int main() {
    cout << " Choose 0 for static camera or 1 for dynamic camera: ";
    int mode;
    cin >> mode;
    if (mode == 0) {
        try {
            vector<pair<string, string>> videos = {
                {"Car Video 1 - Intersection\n", "../Videos/CarVideo1.mp4"},
                {"Car Video 2 - Road\n",         "../Videos/CarVideo2.mp4"},
                {"Car Video 3 - Highway\n",      "../Videos/CarVideo3.mp4"}
            };

            cout << "Select a video: ";
            for (size_t i = 0; i < videos.size(); ++i)
                cout << i + 1 << ". " << videos[i].first << " ";

            int choice;
            cin >> choice;
            if (!cin || choice < 1 || choice > int(videos.size()))
                throw runtime_error("Invalid selection");

            string path = videos[choice - 1].second;
            Mat staticBg = captureBackground(path);
            imshow("Captured Background", staticBg);
            waitKey(1000);

            processStream(path, staticBg);

        } catch (const exception &e) {
            cerr << "Error: " << e.what() << " ";
            return -1;
        }

    } else {
        try {
            vector<pair<string, string>> videos = {
                {"Drone - Traffic 1",       "../Videos/Drone.mp4"},
                {"Drone - Traffic 2",       "../Videos/Drone_2.mp4"}
            };

            cout << "Select a video for dynamic test:\n";
            for (size_t i = 0; i < videos.size(); ++i)
                cout << i + 1 << ". " << videos[i].first << "\n";

            int choice;
            cin >> choice;
            if (!cin || choice < 1 || choice > int(videos.size()))
                throw runtime_error("Invalid selection");

            string path = videos[choice - 1].second;
            VideoCapture cap(path);
            if (!cap.isOpened())
                throw runtime_error("Cannot open video: " + path);

            double fps = cap.get(CAP_PROP_FPS);
            Size frameSize(1280, 720);
            VideoWriter writerOverlay("overlay_output.mp4",
                VideoWriter::fourcc('m','p','4','v'), fps, frameSize);

            Mat prevFrame, prevGray;
            cap >> prevFrame;
            if (prevFrame.empty())
                throw runtime_error("Empty first frame");
            resize(prevFrame, prevFrame, frameSize);
            cvtColor(prevFrame, prevGray, COLOR_BGR2GRAY);

            while (true) {
                Mat frame;
                if (!cap.read(frame) || frame.empty())
                    break;

                resize(frame, frame, frameSize);
                Mat gray;
                cvtColor(frame, gray, COLOR_BGR2GRAY);

                Mat stabFrame = frame;
                Mat stabGray;
                cvtColor(stabFrame, stabGray, COLOR_BGR2GRAY);

                // optical flow & mask
                Mat flowResid  = hornSchunck(prevGray, stabGray, 1.0f, 50);
                Mat movingMask = detectMovingMask(flowResid, 1.5f);

                // for visualization of the algorithm
                Mat thickMask;
                int dilateSize = 17;
                Mat element = getStructuringElement(MORPH_ELLIPSE, Size(dilateSize, dilateSize));
                dilate(movingMask, thickMask, element);
                GaussianBlur(thickMask, thickMask, Size(9,9), 0);

                Mat overlayFrame = stabFrame.clone();
                overlayFrame.setTo(Scalar(0x4F, 0x57, 0x6B), thickMask); // #6B574F

                Mat removedFrame = removeMovingObjects(stabFrame, thickMask, 14);

                Mat blurredFrame;
                GaussianBlur(removedFrame, blurredFrame, Size(27,27), 0);

                Mat finalFrame = removedFrame.clone();
                for (int y = 0; y < finalFrame.rows; ++y) {
                    for (int x = 0; x < finalFrame.cols; ++x) {
                        if (thickMask.at<uchar>(y, x) > 128) {
                            finalFrame.at<Vec3b>(y, x) = blurredFrame.at<Vec3b>(y, x);
                        }
                    }
                }

                writerOverlay.write(finalFrame);

                imshow("Stabilized Frame", stabFrame);
                imshow("Moving Mask",     movingMask);
                imshow("Thick Mask",      thickMask);
                imshow("Overlay Frame",   overlayFrame);
                imshow("Final Blurry Removed", finalFrame);

                if ((char)waitKey(30) == 27) break;
                prevGray = stabGray.clone();
            }

            writerOverlay.release();

        } catch (const exception &e) {
            cerr << "Error (dynamic): " << e.what() << "\n";
            return -1;
        }
    }
    return 0;
}
