//
//  particle_filter.cpp
//  particle-filter
//
//  Created by tomiya on 2017/08/01.
//  Copyright © 2017年 tomiya. All rights reserved.
//

#include "particle_filter.hpp"

namespace pf {
    
    void init(vector<Particle>& particles, int n_particles, double center_x, double center_y){
        for(int i = 0; i < n_particles; i++){
            Particle particle;
            particle.x = center_x;
            particle.y = center_y;
            particle.weight = 1.0 / n_particles;
            particles.push_back(particle);
        }
    }
    
    void resample(vector<Particle>& particles){
        vector<Particle> tmp_particles;
        tmp_particles = particles;
        vector<double> weights;
        weights.push_back(particles[0].weight);
        for(int i = 1; i < particles.size(); i++){
            weights.push_back(weights[i-1] + particles[i].weight);
        }
        double last_weight = weights.back();
        random_device rnd;
        mt19937 mt(rand());
        uniform_real_distribution<> rand(0.0, 1.0);
        for(int i = 0; i < particles.size(); i++){
            double weight = rand(mt) * last_weight;
            for(int j = 0; j < weights.size(); j++){
                if(weight > weights[j]){
                    continue;
                }
                else{
                    particles[i].x = tmp_particles[j].x;
                    particles[i].y = tmp_particles[j].y;
                    particles[i].weight = 1.0;
                    break;
                }
            }
        }
    }
    
    void predict(vector<Particle>& particles, double variance){
        random_device rnd;
        default_random_engine engine(rnd());
        normal_distribution<> dist(0.0, sqrt(variance));
        for(int i = 0; i < particles.size(); i++){
            particles[i].x += dist(engine);
            particles[i].y += dist(engine);
        }
    }
    
    void weight(vector<Particle>& particles, Mat image, double (*likelihood)(int, int, Mat)){
        double sum_weight = 0.0;
        for(int i = 0; i < particles.size(); i++){
            particles[i].weight = likelihood(particles[i].x, particles[i].y, image);
            sum_weight += particles[i].weight;
        }
        for(int i = 0; i < particles.size(); i++)
            particles[i].weight /= sum_weight;
    }
    
    void measure(vector<Particle> particles, double& center_x, double& center_y){
        double x = 0.0;
        double y = 0.0;
        double weight = 0.0;
        for(int i = 0; i < particles.size(); i++){
            x += particles[i].x * particles[i].weight;
            y += particles[i].y * particles[i].weight;
            weight += particles[i].weight;
        }
        center_x = x / weight;
        center_y = y / weight;
    }
}
