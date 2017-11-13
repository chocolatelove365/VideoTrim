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
#define SUBTRACT 1

using namespace std;
using namespace cv;
using json = nlohmann::json;

string video_path, background_path;
VideoCapture cap;
Mat background, g_background;
Rect range;

void load_config(const char *path){
    ifstream input_json(path);
    if(input_json.is_open()){
        json j;
        input_json >> j;
        video_path = j["video_path"];
        background_path = j["background_path"];
    }
//    range.x = 500;
//    range.y = 300;
//    range.width = 1920;
//    range.height = 1080;
    range.x = 0;
    range.y = 0;
    range.width = 3840;
    range.height = 2160;
}

void init(){
    load_config("config.json");
    
    cap = VideoCapture(video_path);
    if(!cap.isOpened()){
        cout << "ERROR: failed to open video.\n";
        return;
    }
    int width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    int height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    cout << "width: " << width << ", height: " << height << "\n";
    
    g_background = imread(background_path, 0);
    threshold(g_background, g_background, 0, 255, CV_THRESH_BINARY|CV_THRESH_OTSU);
}

int main(int argc, const char * argv[]) {
    Mat frame, g_frame, diff;
#if TRIM_VIDEO
    Mat trim_frame, small_trim_frame;
#else
    Mat small_frame;
#endif
    
    init();
    namedWindow("VideoTrim", 1);
    
    while(waitKey(1) != 'q'){
        cap >> frame;
        if(frame.empty()){
            cout << "ERROR: failed to capture video.\n";
            break;
        }
        cvtColor(frame, g_frame, CV_BGR2GRAY);
        threshold(g_frame, g_frame, 0, 255, CV_THRESH_BINARY|CV_THRESH_OTSU);
        absdiff(g_frame, g_background, diff);
#if TRIM_VIDEO
        trim_frame = Mat(frame, range);
        resize(trim_frame, small_trim_frame, Size(trim_frame.cols/4, trim_frame.rows/4));
        imshow("VideoTrim", small_trim_frame);
#else
        Point pt1(range.x, range.y);
        Point pt2(range.x + range.width, range.y + range.height);
        rectangle(diff, pt1, pt2, 255, 10);
//        resize(frame, small_frame, Size(frame.cols/8, frame.rows/8));
        resize(diff, small_frame, Size(frame.cols/8, frame.rows/8));
        imshow("VideoTrim", small_frame);
#endif
    }
    
    return 0;
}
