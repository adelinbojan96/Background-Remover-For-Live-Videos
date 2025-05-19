#include "static_camera.h"
using namespace std;
using namespace cv;
using namespace dnn;
int main() {
    try {
        vector<pair<string, string>> videos = {
            {"Car Video 1 - Intersection", "../Videos/CarVideo1.mp4"},
            {"Car Video 2 - Road", "../Videos/CarVideo2.mp4"},
            {"Car Video 3 - Highway", "../Videos/CarVideo3.mp4"}
        };

        cout << "Select a video:\n";
        for (size_t i = 0; i < videos.size(); ++i)
            cout << i + 1 << ". " << videos[i].first << "\n";

        int choice;
        cin >> choice;
        if (!cin || choice < 1 || choice > int(videos.size()))
            throw runtime_error("Invalid selection");

        string path = videos[choice - 1].second;
        Mat staticBg;
        cout << "Capturing static background...\n";
        staticBg = captureBackground(path);
        imshow("Captured Static Background", staticBg);
        waitKey(1000);

        processStream(path, staticBg);
    } catch (const exception &e) {
        cerr << "Error: " << e.what() << "\n";
        return -1;
    }
    return 0;
}
