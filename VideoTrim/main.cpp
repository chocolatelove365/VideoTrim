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
#include "particle_filter.hpp"

#define VIDEO_ID 2
#define TRIM_VIDEO 1
#define SCALE 0.5

using namespace std;
using namespace cv;
using json = nlohmann::json;

string video_path, background_path;
VideoCapture cap;
Mat background;
vector<Mat> hsv_backgrounds;
Rect range;
vector<pf::Particle> particles;

void load_config(const char *path){
    ifstream input_json(path);
    if(input_json.is_open()){
        json j;
        input_json >> j;
        video_path = j["video"][VIDEO_ID]["video_path"];
        background_path = j["video"][VIDEO_ID]["background_path"];
    }
    range.x = 500;
    range.y = 300;
    range.width = 1920/2;
    range.height = 1080/2;
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
    
    background = imread(background_path);
    resize(background, background, Size(background.cols/2, background.rows/2));
    Mat tmp;
    cvtColor(background, tmp, CV_BGR2HSV);
    split(tmp, hsv_backgrounds);
    
    pf::init(particles, 500, 1980/2, 1080/2);
}

double likelihood(int x, int y, Mat image){
    int gry = (int)image.at<unsigned char>(y, x);
    return (double)gry > 0 ? 1 : 0.01;
}

void bgr2hsv(float b, float g, float r, float &h, float &s, float &v){
    const float min = std::min(std::min(b, g), r);
    const float max = std::max(std::max(b, g), r);
    h = 0.f;
    s = 0.f;
    v = max;
    const float delta = max - min;
    if(delta != 0.f){
        s = delta / max;
        if(r == max){
            h = (g - b) / delta;
        }
        else if(g == max){
            h = 2.f + (b - r) / delta;
        }
        else{
            h = 4.f + (r - g) / delta;
        }
        h /= 6.f;
        if(h < 0.f) h += 1.f;
    }
}
double likelihood_color(int x, int y, Mat image){
    const float s = image.at<unsigned char>(y, x);
    return s > 50 ? 1 : 0.0001;
}

void onMouse(int event, int x, int y, int, void*){
    switch(event){
        case EVENT_LBUTTONDOWN:
            x /= SCALE;
            y /= SCALE;
            for(int i = 0; i < particles.size(); i++){
#if TRIM_VIDEO
                particles[i].x = x + range.x;
                particles[i].y = y + range.y;
#else
                particles[i].x = x;
                particles[i].y = y;
#endif
            }
            break;
    }
}

int main(int argc, const char * argv[]) {
    Mat frame, diff;
#if TRIM_VIDEO
    Mat trim_frame, small_trim_frame;
#else
    Mat small_frame;
#endif
    
    init();
    namedWindow("VideoTrim", 1);
    setMouseCallback("VideoTrim", onMouse);
    
    while(waitKey(1) != 'q'){
        cap >> frame;
        if(frame.empty()){
            cout << "ERROR: failed to capture video.\n";
            break;
        }
        resize(frame, frame, Size(frame.cols/2, frame.rows/2));
        Mat tmp;
        cvtColor(frame, tmp, CV_BGR2HSV);
        vector<Mat> hsv_planes;
        split(tmp, hsv_planes);
        absdiff(hsv_planes[1], hsv_backgrounds[1], diff);
        threshold(diff, diff, 0, 255, CV_THRESH_BINARY|CV_THRESH_OTSU);
        
        // Update particles
        pf::resample(particles);
        pf::predict(particles);
        pf::weight(particles, diff, likelihood_color);
        double center_x;
        double center_y;
        pf::measure(particles, center_x, center_y);
        
        // Draw
        for(int i = 0; i < particles.size(); i++){
            int x = (int)particles[i].x;
            int y = (int)particles[i].y;
            circle(diff, Point(x, y), 5, 255, -1);
        }
        
        range.x = (int)(center_x - range.width/2);
        range.y = (int)(center_y - range.height/2);
        range.x = std::min(std::max(0, range.x), frame.cols - range.width);
        range.y = std::min(std::max(0, range.y), frame.rows - range.height);
#if TRIM_VIDEO
        trim_frame = Mat(frame, range);
        resize(trim_frame, trim_frame, Size(trim_frame.cols * SCALE, trim_frame.rows * SCALE));
        imshow("VideoTrim", trim_frame);
#else
        Point pt1(range.x, range.y);
        Point pt2(range.x + range.width, range.y + range.height);
        rectangle(diff, pt1, pt2, 255, 5);
        resize(diff, diff, Size(diff.cols * SCALE, diff.rows * SCALE));
        imshow("VideoTrim", diff);
#endif
    }
    
    return 0;
}
