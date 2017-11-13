//
//  main.cpp
//  VideoTrim
//
//  Created by tomiya on 2017/11/13.
//  Copyright © 2017年 tomiya. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include "json.hpp"

#define TRIM_VIDEO 0

using namespace std;
using namespace cv;
using json = nlohmann::json;

string video_path;
VideoCapture cap;
Rect range;

void load_config(const char *path){
    ifstream input_json(path);
    if(input_json.is_open()){
        json j;
        input_json >> j;
        video_path = j["video_path"];
    }
#if TRIM_VIDEO
    range.x = 0;
    range.y = 0;
    range.width = 1920;
    range.height = 1080;
#endif
}

int main(int argc, const char * argv[]) {
    Mat frame;
#if TRIM_VIDEO
    Mat trim_frame, small_trim_frame;
#else
    Mat small_frame;
#endif
    
    load_config("config.json");
    
    cap = VideoCapture(video_path);
    if(!cap.isOpened()){
        cout << "ERROR: failed to open video.\n";
        return -1;
    }
    int width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    int height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    cout << "width: " << width << ", height: " << height << "\n";
    
    namedWindow("VideoTrim", 1);
    
    while(waitKey(1) != 'q'){
        cap >> frame;
        if(frame.empty()){
            cout << "ERROR: failed to capture video.\n";
            break;
        }
#if TRIM_VIDEO
        trim_frame = Mat(frame, range);
        resize(trim_frame, small_trim_frame, Size(trim_frame.cols/4, trim_frame.rows/4));
        imshow("VideoTrim", small_trim_frame);
#else
        resize(frame, small_frame, Size(frame.cols/8, frame.rows/8));
        imshow("VideoTrim", small_frame);
#endif
    }
    
    return 0;
}
