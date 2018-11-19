#pragma once
#include "samplers/poissonsampler.h"

#include "eigen/Eigen/Dense"

class PoissonSampler;
class Particle;

class ParticleGrid
{
public:
    // constructors
    ParticleGrid(PoissonSampler* samp);

    // methods
    void p2gTransfer();
    void applyForces();
    void g2pTransfer();
    void particleAdvection();

    float computeWeight(glm::vec3 evalPos, glm::vec3 gridPos);
    glm::vec3 computeWeightGrad(glm::vec3 evalPos, glm::vec3 gridPos);

    // fields
    std::vector<std::vector<std::vector<Particle*>>> nodes;
    PoissonSampler* sampler;
    std::vector<float> gridMasses;
    std::vector<glm::vec3> gridVelocities;
    std::vector<glm::vec3> gridMomentums;
    std::vector<glm::vec3> gridForces;

    // need to do: initialize a bunch of things..
    // i think the math and stuff is right but idk what values to start this shit
};
