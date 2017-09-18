/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

  num_particles = 100;

  // default_random_engine gen;

  weights.resize(num_particles);
  particles.resize(num_particles);


  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; ++i)
  {
    particles[i].id = i;
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
    particles[i].weight = 1.0;
  }

  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

  // default_random_engine gen;

  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  for (int i = 0; i < num_particles; ++i)
  {
    if (fabs(yaw_rate) < 0.0001)
    {
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    }
    else
    {
      particles[i].x += (velocity / yaw_rate) * ((sin(particles[i].theta + yaw_rate * delta_t)
                                                - sin(particles[i].theta)));
      particles[i].y += (velocity / yaw_rate) * ((cos(particles[i].theta)
                                                - cos(particles[i].theta + yaw_rate * delta_t)));
      particles[i].theta += yaw_rate * delta_t;
    }

    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
  
  for (unsigned int i = 0; i < observations.size(); ++i)
  {
    LandmarkObs obs = observations[i];

    // cout << obs.x << obs.y << obs.id << endl;

    double default_dist = numeric_limits<double>::max();

    int nearest_id = -1;
    
    for (int k = 0; k < predicted.size(); ++k)
    {
      LandmarkObs pred = predicted[k];

      // cout << pred.x << pred.y << pred.id << endl;

      double meas_dist = dist(obs.x, obs.y, pred.x, pred.y);

      if (meas_dist < default_dist)
      {
        nearest_id = pred.id;
        default_dist = meas_dist;
        // cout << "New close object!" << endl;
      }
    }

    observations[i].id = nearest_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

  for (int i = 0; i < num_particles; ++i)
  {
    double px = particles[i].x;
    double py = particles[i].y;
    double ptheta = particles[i].theta;

    vector<LandmarkObs> pred_landmarks;
    for (unsigned int k = 0; k < map_landmarks.landmark_list.size(); ++k)
    {
      if (dist(px, py, map_landmarks.landmark_list[k].x_f, map_landmarks.landmark_list[k].y_f) < sensor_range)
      {
        pred_landmarks.push_back(LandmarkObs{map_landmarks.landmark_list[k].id_i, map_landmarks.landmark_list[k].x_f, map_landmarks.landmark_list[k].y_f});
      }
    }

    vector<LandmarkObs> transformed;
    for (unsigned int k = 0; k < observations.size(); ++k)
    {
      double tx = cos(ptheta) * observations[k].x - sin(ptheta) * observations[k].y + px;
      double ty = sin(ptheta) * observations[k].x + cos(ptheta) * observations[k].y + py;
      transformed.push_back(LandmarkObs{observations[k].id, tx, ty});
    }

    dataAssociation(pred_landmarks, transformed);

    particles[i].weight = 1.0;

    for (unsigned int k = 0; k < transformed.size(); ++k)
    {
      double ox = transformed[k].x;
      double oy = transformed[k].y;
      int oi = transformed[k].id;

      // LandmarkObs extracted = std::find_if(predicted.begin(), predicted.end(),
      //                                      boost::bind(&Landmarkobs::id, _1) == oi);

      // double mx = extracted.x;
      // double my - extracted.y;

      double mx;
      double my;

      for (unsigned int j = 0; j < pred_landmarks.size(); ++j)
      {
        if (pred_landmarks[j].id == oi)
        {
          mx = pred_landmarks[j].x;
          my = pred_landmarks[j].y;
        }
      }

      double c1 = 1.0 / (2.0 * M_PI * std_landmark[0] * std_landmark[1]);
      double c2 = pow(ox - mx, 2) / pow(std_landmark[0], 2);
      double c3 = pow(oy - my, 2) / pow(std_landmark[1], 2);

      double weight = c1 * exp(-0.5 * (c2 + c3));

      if (weight < 0.0001)
      {
        weight = 0.0001;
      }

      particles[i].weight *= weight;

    }

  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  vector<Particle> new_particles;

  double max_weight = 0.0;
  // vector<double> weights;
  for (int i = 0; i < num_particles; ++i)
  {
    weights[i] = particles[i].weight;
    if (particles[i].weight > max_weight)
    {
      max_weight = particles[i].weight;
    }
  }

  uniform_int_distribution<int> resampledist(0, num_particles-1);
  auto resample_index = resampledist(gen);

  uniform_real_distribution<double> weightdist(0, max_weight);

  double beta = 0.0;

  for (int i = 0; i < num_particles; ++i)
  {
    beta += weightdist(gen) * 2.0;

    while (weights[resample_index] < beta)
    {
      beta -= weights[resample_index];
      resample_index = (resample_index + 1) % num_particles;
    }

    new_particles.push_back(particles[resample_index]);
  }

  particles = new_particles;

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
