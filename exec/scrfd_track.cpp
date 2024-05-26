#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <chrono>
#include "BYTETracker.h"
#include "SCRFD.h"
#include "model_load.h"

using namespace cv;
using namespace std;
int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [videopath]\n", argv[0]);
        return -1;
    }

    const char* videopath = argv[1];
    VideoCapture cap(videopath);
    if (!cap.isOpened())
        return 0;

    int img_w = cap.get(CAP_PROP_FRAME_WIDTH);
    int img_h = cap.get(CAP_PROP_FRAME_HEIGHT);
    int fps = cap.get(CAP_PROP_FPS);
    long nFrame = static_cast<long>(cap.get(CAP_PROP_FRAME_COUNT));
    cout << "Total frames: " << nFrame << ", fps: " << fps << endl;

    VideoWriter writer("demo.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(img_w, img_h));

    Mat img;
    SCRFD scrfd;
    string model_path = "/Users/tunm/work/mnn-scrfd/models/scrfd_2.5g_bnkps_shape320x320.mnn";
    scrfd.load_heads(scrfd_2_5g_bnkps_head_info);
    scrfd.reload(model_path, true, 640, 2, 1);

    BYTETracker tracker(fps, 30);
    int num_frames = 0;
    int total_ms = 1;
    for (;;)
    {
        if (!cap.read(img))
            break;
        num_frames++;
        
        if (num_frames % 20 == 0)
        {
            cout << "Processing frame " << num_frames << " (" << num_frames * 1000000 / total_ms << " fps)" << endl;
        }
        if (img.empty())
            break;

        vector<FaceInfo> results;
        auto start = chrono::system_clock::now();
        scrfd.detect(img, results);
        vector<Object> objects;
        for (const auto& face : results) {
            Object obj;
            obj.rect = Rect_<float>(face.x1, face.y1, face.x2 - face.x1, face.y2 - face.y1);
            obj.label = 0; // assuming all detections are faces
            obj.prob = face.score;
            objects.push_back(obj);
        }
        vector<STrack> output_stracks = tracker.update(objects);
        auto end = chrono::system_clock::now();
        total_ms = total_ms + chrono::duration_cast<chrono::microseconds>(end - start).count();
        for (int i = 0; i < output_stracks.size(); i++)
        {
            vector<float> tlwh = output_stracks[i].tlwh;
            bool vertical = tlwh[2] / tlwh[3] > 1.6;
            if (tlwh[2] * tlwh[3] > 20 && !vertical)
            {
                Scalar s = tracker.get_color(output_stracks[i].track_id);
                putText(img, format("%d", output_stracks[i].track_id), Point(tlwh[0], tlwh[1] - 5), 
                        0, 0.6, Scalar(0, 0, 255), 2, LINE_AA);
                rectangle(img, Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);
            }
        }
        putText(img, format("frame: %d fps: %d num: %d", num_frames, num_frames * 1000000 / total_ms, (int)output_stracks.size()),
                Point(0, 30), 0, 0.6, Scalar(0, 0, 255), 2, LINE_AA);
        writer.write(img);
    }
    cap.release();
    cout << "FPS: " << num_frames * 1000000 / total_ms << endl;

    return 0;
}
