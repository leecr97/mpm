#pragma once
#include "samplers/poissonsampler.h"

#include "eigen/Eigen/Dense"

class PoissonSampler;
class Particle;

class ParticleGrid
{
public:
    // constructors
    ParticleGrid(PoissonSampler* sampler);
    ~ParticleGrid() { delete sampler;
                      gridMasses.clear(); gridVelocities.clear();
                      gridMomentums.clear(); gridForces.clear(); }

    // methods
    void initialize();
    void MPM();
    void computeWeights();
    void p2gTransfer();
    void computeGridVelocities();
    bool collisionCheck(glm::vec3 loc);
    void computeForces();
    void velocityUpdate();
    void updateDGs();
    void g2pTransfer();
    void particleAdvection();

    float weightFunc(glm::vec3 evalPos, glm::vec3 gridInd);
    glm::vec3 weightGradFunc(glm::vec3 evalPos, glm::vec3 gridInd);
    glm::vec3 posToGrid(glm::vec3 p);
    glm::vec3 gridToPos(int locx, int locy, int locz);
    void reset();

    // fields
    PoissonSampler* sampler;
    glm::vec3 gridDim;
    float cellSize;
    float dt = 1e-6;
    Bounds3f gridBounds;
    std::vector<std::vector<std::vector<float>>> gridMasses;
    std::vector<std::vector<std::vector<glm::vec3>>> gridVelocities;
    std::vector<std::vector<std::vector<glm::vec3>>> gridMomentums;
    std::vector<std::vector<std::vector<glm::vec3>>> gridForces;

    // need to do: initialize a bunch of things..
    // i think the math and stuff is right but idk what values to start this shit
};
