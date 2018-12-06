#pragma once
#include "scene/scene.h"
#include "scene/geometry/mesh.h"
#include "raytracing/ray.h"
#include "globals.h"
#include <iostream>

#include "eigen/Eigen/Dense"

#include "poissonbvh.h"
#include "openGL/drawable.h"

typedef Eigen::Matrix<float, 3, 3> EigenMat3;
typedef Eigen::Vector3d EigenVec3;

class Particle;

class PoissonSampler : public Drawable
{
public:
    //assuming only inputting primitives in scene that ALL need to be filled
    PoissonSampler(Mesh& mesh, Scene& scene);
    ~PoissonSampler() { finalSamples.clear(); origPositions.clear(); delete bvh; delete bounds;}

    // all variables below are initialized in the constructor's list
    Mesh m;
    Scene s;

    // extent of the sample domain in Rn is 2d or 3d
//    bool threeDim;
    int nDim = 3;
    // bounds of the mesh
    Bounds3f* bounds;
    // dimensions of the grid (# of voxels)
    glm::vec3 gridDim;
    // cell size (r/√n)
    float cellSize;
    PoissonBVH* bvh;
    // minimum distance between samples
//    float radius = 0.4f;
    float radius = 0.3f * sqrt(3.0f);
    // limit of samples to choose before rejection in the algorithm
    int K = 30;

    void initialize();
    std::vector<Particle> finalSamples;
    std::vector<glm::vec3> origPositions;
    Sampler samp;
    int numPoints;

    void generateSamples();
    glm::vec3 posToGridLoc(glm::vec3 p);
    bool validWithinBounds(glm::vec3 p);
    glm::vec3 randomLocAround(glm::vec3 pos);

    virtual GLenum drawMode() const;
    virtual void create();

    void reset();
};

class Particle {
    public:
        Particle(glm::vec3 gLoc, glm::vec3 wPos, int gId)
            : gridLoc(gLoc), pos(wPos), id(gId),
              mass(1.f), vol(1.f), vp(glm::vec3(0.f)), Bp(glm::vec3(0.f)),
              elasticity(glm::mat3(1.f)), plasticity(glm::mat3(1.f)),
              stress(glm::mat3(1.f)), weight(0.0f), weightGrad(glm::vec3(0.0f)) {}
        Particle(Particle* s)
            : gridLoc(s->gridLoc), pos(s->pos), id(s->id),
              mass(1.f), vol(1.f), vp(glm::vec3(0.f)), Bp(glm::vec3(0.f)),
              elasticity(glm::mat3(1.f)), plasticity(glm::mat3(1.f)),
              stress(glm::mat3(1.f)), weight(0.0f), weightGrad(glm::vec3(0.0f)) {}
        ~Particle() {}

        // position in world
        glm::vec3 pos;
        // position on grid (grid used in both sampling and mpm)
        glm::vec3 gridLoc;
        int id;

        // particle info
        // mass (mp)
        float mass;
        // volume (Vp)
        float vol;
        // initial velocity
        glm::vec3 vp;
        // affine matrix (type?)
        glm::vec3 Bp;
        // elasticity (Fe)
        glm::mat3 elasticity;
        // plasticity (Fp)
        glm::mat3 plasticity;
        // stress (piola-kirchoff or cauchy idk)
        glm::mat3 stress;
        // weight (for grid stuff)
        float weight;
        // weight gradient (also for grid stuff)
        glm::vec3 weightGrad;

        void update(glm::mat3 newDG);

        // deformation gradient (F = FeFp)
        glm::mat3 deformationGrad() {
            return (elasticity * plasticity);
        }
};
