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

#define TRIM_VIDEO 1
#define SUBTRACT 1

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
        video_path = j["video_path"];
        background_path = j["background_path"];
    }
    range.x = 500;
    range.y = 300;
    range.width = 1920;
    range.height = 1080;
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
//    resize(background, background, Size(background.cols/8, background.rows/8));
    Mat tmp;
    cvtColor(background, tmp, CV_BGR2HSV);
    split(tmp, hsv_backgrounds);
    
    pf::init(particles, 500, 1980, 1080);
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
//    const float b = image.at<Vec3b>(y, x)[0];
//    const float g = image.at<Vec3b>(y, x)[1];
//    const float r = image.at<Vec3b>(y, x)[2];
//    float h, s, v;
//    bgr2hsv(b, g, r, h, s, v);
//    h *= 360;
//    return h > 220 && h < 260 ? 1 : 0.01;
    return s > 200 ? 1 : 0.01;
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
    
    while(waitKey(1) != 'q'){
        cap >> frame;
        if(frame.empty()){
            cout << "ERROR: failed to capture video.\n";
            break;
        }
        
        Mat tmp;
        cvtColor(frame, tmp, CV_BGR2HSV);
        vector<Mat> hsv_planes;
        split(tmp, hsv_planes);
        absdiff(hsv_planes[1], hsv_backgrounds[1], diff);
        
        // Update particles
        pf::resample(particles);
        pf::predict(particles);
        pf::weight(particles, diff, likelihood);
        double center_x;
        double center_y;
        pf::measure(particles, center_x, center_y);
        
        // Draw
        for(int i = 0; i < particles.size(); i++){
            int x = (int)particles[i].x;
            int y = (int)particles[i].y;
            circle(diff, Point(x, y), 10, 255, -1);
        }
        
        range.x = (int)(center_x - range.width/2);
        range.y = (int)(center_y - range.height/2);
        range.x = std::min(std::max(0, range.x), frame.cols - range.width);
        range.y = std::min(std::max(0, range.y), frame.rows - range.height);
#if TRIM_VIDEO
        trim_frame = Mat(frame, range);
//        trim_frame = Mat(diff, range);
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
