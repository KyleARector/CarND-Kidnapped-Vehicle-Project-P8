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

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Instantiate a random generator and resize weights and particle vector
	num_particles = 100;
	weights.resize(num_particles, 1.0);
	particles.resize(num_particles);

	// Instantiate a random generator
	default_random_engine gen;

	// Create normal distributions for x, y, and theta
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	// For each particle, sample from the normal distributions about the GPS coordinates
	for (int i = 0; i < num_particles; ++i) {
		// Id is the position in the vector
		particles[i].id = i;
		// Initialize weight at 1
		particles[i].weight = 1.0;
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
	}
	// Set flag to true to prevent duplicate initialization
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Instantiate a random generator
	default_random_engine gen;

  // Create normal distributions for noise
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  // Cycle through particles
	for (int i = 0; i < num_particles; ++i) {
		// Determine if the yaw is changing over time
  	if (abs(yaw_rate) != 0) {
      particles[i].x += (velocity/yaw_rate) * (sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta));
      particles[i].y += (velocity/yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate * delta_t)));
      particles[i].theta += yaw_rate * delta_t;
    }
		// If yaw rate is static, use simplified equations
		else {
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    }
    // Add gaussian noise
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
		particles[i].theta += dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Cycle through observations
	for (int i = 0; i < observations.size(); ++i) {
		// Set initially large min distance to compare to
		double min_distance = 9999999.9;
		// Cycle through predection landmarks
		for (int j = 0; j < predicted.size(); j++) {
			// Check the distance between the observation and predicted landmark
			double distance = dist(predicted[j].x,
				                     predicted[j].y,
												     observations[i].x,
												     observations[i].y);
			// Compare the distance to the minimum distance, and replace if smaller
			if (distance < min_distance) {
				// Update the min distance
				min_distance = distance;
				// Set the observations landmark id to the loop counter
				observations[i].id = j;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// Break values out for ease of writing
	double std_x = std_landmark[0];
	double std_y = std_landmark[1];
	double c = 1.0/(2.0 * M_PI * std_x * std_y);

	// Cycle through particles
	for (int i = 0; i < num_particles; ++i) {
		// Break values out for ease of writing
		double p_x = particles[i].x;
		double p_y = particles[i].y;
		double p_theta = particles[i].theta;
		particles[i].weight = 1.0/num_particles;

		// Create a vector for in work landmarks
		vector<LandmarkObs> in_work_lm;
		vector<LandmarkObs> trans_obs;

		// Loop through all landmarks to see which are in range
		for (int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
			float x = map_landmarks.landmark_list[j].x_f;
			float y = map_landmarks.landmark_list[j].y_f;
			int id = map_landmarks.landmark_list[j].id_i;
			// If the distance to the landmark is less than the sensor range, add it to the vector
			if (dist(x, y, p_x, p_y) <= sensor_range) {
				// Add a new landmark to the vector with values calculated above
				in_work_lm.push_back(LandmarkObs{id, x, y});
			}
		}

		// Apply transformation to map coordinates
		for (int j = 0; j < observations.size(); ++j) {
			double trans_x = p_x + (cos(p_theta) * observations[j].x) - (sin(p_theta) * observations[j].y);
			double trans_y = p_y + (sin(p_theta) * observations[j].x) + (cos(p_theta) * observations[j].y);
			// Add to the translated list
			trans_obs.push_back(LandmarkObs{observations[j].id, trans_x, trans_y});
		}

		// Associated observations with nearest landmarks
		dataAssociation(in_work_lm, trans_obs);
		double calc_weight = 1.0;

		for (int j = 0; j < trans_obs.size(); ++j) {
			int obs_id = trans_obs[j].id;
			double p_x = trans_obs[j].x;
			double p_y = trans_obs[j].y;

			double lm_x = in_work_lm[obs_id].x;
			double lm_y = in_work_lm[obs_id].y;

			// Calculate weights
			double upper = (pow((p_x - lm_x), 2)/(2.0 * std_x * std_x)) +
			               (pow((p_y - lm_y), 2)/(2.0 * std_y * std_y));
			double p = c*exp(-upper);
			calc_weight *= p;
		}
		particles[i].weight = calc_weight;
		weights[i] = calc_weight;
	}
}

void ParticleFilter::resample() {
	// Create a temporary vector for new particles with the same size as particles
	vector<Particle> temp_particles(num_particles);

	// Instantiate a random generator
	default_random_engine gen;

	// Cycle through particles
	for (int i = 0; i < num_particles; ++i) {
			// Create a discrete_distribution over the weights vector
      discrete_distribution<int> d_dist(weights.begin(), weights.end());
			// Add a particle to the temp set based on the result of the discrete_distribution
      temp_particles[i] = particles[d_dist(gen)];
	}

	// Replace old particles with new set
	particles = temp_particles;
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
