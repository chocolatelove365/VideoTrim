//
//  particle_filter.hpp
//  particle-filter
//
//  Created by tomiya on 2017/08/01.
//  Copyright © 2017年 tomiya. All rights reserved.
//

#ifndef particle_filter_hpp
#define particle_filter_hpp

#include <stdio.h>
#include <vector>
#include <random>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

namespace pf {
    
    struct Particle{
        double x;
        double y;
        double weight;
    };
    
    double likelihood(int x, int y, Mat image);
    void init(vector<Particle>& particles, int n_particles = 100, double center_x = 0.0, double center_y = 0.0);
    void resample(vector<Particle>& particles);
    void predict(vector<Particle>& particles, double variance = 60.0);
    void weight(vector<Particle>& particles, Mat image, double (*likelihood)(int, int, Mat));
    void measure(vector<Particle> particles, double& center_x, double& center_y);
}

#endif /* particle_filter_hpp */
