#include "static_camera.h"
#include "dynamic_camera.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <utility>
using namespace std;
using namespace cv;
using namespace dnn;


int main() {
    cout << "Choose 1 for static camera or 2 for dynamic camera: ";
    int mode;
    cin >> mode;
    if (mode == 1) {
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
        cout<<"Program will use all available CPU threads for processing (" <<getNumberOfCPUs()<<")\n";
        setNumThreads(getNumberOfCPUs());
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

                Mat gray(frame.size(), CV_8UC1);

                for (int y = 0; y < frame.rows; ++y)
                    for (int x = 0; x < frame.cols; ++x) {
                        Vec3b bgr = frame.at<Vec3b>(y, x);
                        gray.at<uchar>(y, x) = uchar(0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2]);
                    }

                Mat stabFrame = frame;
                Mat stabGray(stabFrame.size(), CV_8UC1);
                for (int y = 0; y < stabFrame.rows; ++y)
                    for (int x = 0; x < stabFrame.cols; ++x) {
                        Vec3b bgr = stabFrame.at<Vec3b>(y, x);
                        stabGray.at<uchar>(y, x) = uchar(0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2]);
                    }

                Mat flowResid  = hornSchunck(prevGray, stabGray, 1.0f, 50);
                Mat movingMask = detectMovingMask(flowResid, 1.5f);

                int dilateSize = 17;
                Mat element = ellipseKernel_2(dilateSize);

                Mat thickMask;
                dilate(movingMask, thickMask, element);

                thickMask = gaussianBlur(thickMask, 17, 5.0f);

                Mat overlayFrame = stabFrame.clone();

                parallel_for_(Range(0, overlayFrame.rows), [&](const Range& r){
                for (int y = r.start; y < overlayFrame.rows; ++y)
                    for (int x = 0; x < overlayFrame.cols; ++x)
                        if (thickMask.at<uchar>(y, x) > 0)
                            overlayFrame.at<Vec3b>(y, x) = Vec3b(0x4F, 0x57, 0x6B);
                });

                Mat removedFrame = removeMovingObjects(stabFrame, thickMask);

                Mat blurredFrame = gaussianBlur(removedFrame, 27, 10.0f);

                // blur the original frame
                Mat maskBlurredFrame = stabFrame.clone();
                parallel_for_(Range(0, maskBlurredFrame.rows), [&](const Range& r) {
                    for (int y = r.start; y < maskBlurredFrame.rows; ++y)
                        for (int x = 0; x < maskBlurredFrame.cols; ++x)
                            if (thickMask.at<uchar>(y, x) > 128)
                                maskBlurredFrame.at<Vec3b>(y, x) = blurredFrame.at<Vec3b>(y, x);
                });
                // blur using thick mask the removed objects frame
                Mat finalFrame = removedFrame.clone();

                parallel_for_(Range(0, finalFrame.rows), [&](const Range& r) {
                    for (int y = r.start; y < finalFrame.rows; ++y) {
                        for (int x = 0; x < finalFrame.cols; ++x) {
                            if (thickMask.at<uchar>(y, x) > 128) {
                                finalFrame.at<Vec3b>(y, x) = blurredFrame.at<Vec3b>(y, x);
                            }
                        }
                    }
                });

                writerOverlay.write(finalFrame);

                imshow("Stabilized Frame", stabFrame);
                imshow("Moving Mask",     movingMask);
                imshow("Thick Mask",      thickMask);
                imshow("Overlay Frame",   overlayFrame);
                imshow("Removed Moving Objects (Using Telea)", removedFrame);
                imshow("Blurred Thick Mask (Using Blur)", maskBlurredFrame);
                imshow("Final Blurry Removed (Telea + Blur)", finalFrame);

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
