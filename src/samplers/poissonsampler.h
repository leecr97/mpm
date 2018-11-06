#pragma once
#include "scene/scene.h"
#include "scene/geometry/mesh.h"
#include "raytracing/ray.h"
#include "globals.h"
#include <iostream>

#include "eigen/Eigen/Dense"

#include "poissonbvh.h"
#include "openGL/drawable.h"

class Particle;

class PoissonSampler : public Drawable
{
public:
    //assuming only inputting primitives in scene that ALL need to be filled
    PoissonSampler(Mesh& mesh, Scene& scene, bool isThreeDim);
    ~PoissonSampler() { finalSamples.clear(); delete bvh; delete bounds;}

    // all variables below are initialized in the constructor's list
    Mesh m;
    Scene s;

    // extent of the sample domain in Rn is 2d or 3d
    bool threeDim;
    // bounds of the mesh
    Bounds3f* bounds;
    // dimensions of the grid (# of voxels)
    glm::vec3 gridDim;
    // cell size (r/√n)
    float cellSize;
    PoissonBVH* bvh;
    // minimum distance between samples
    float radius = 0.2f;
    // limit of samples to choose before rejection in the algorithm
    int K;

    void initialize();
    std::vector<Particle*> finalSamples;
    std::vector<glm::vec3> origPositions;
    Sampler samp;
    int numPoints;

    void generateSamples();
    glm::vec3 posToGridLoc(glm::vec3 p);
    bool validWithinBounds(glm::vec3 p);
    glm::vec3 randomLocAround(glm::vec3 pos);

    virtual GLenum drawMode() const;
    virtual void create();

    void fallWithGravity();
    void resetParticlePositions();
};

class Particle {
    public:
        Particle(glm::vec3 gLoc, glm::vec3 wPos, int gId)
            : gridLoc(gLoc), pos(wPos), id(gId) {}
        Particle(Particle* s)
            : gridLoc(s->gridLoc), pos(s->pos), id(s->id) {}
        ~Particle();

        glm::vec3 pos;
        glm::vec3 gridLoc;
        int id;
};
