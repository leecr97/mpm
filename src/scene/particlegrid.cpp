#include "particlegrid.h"


ParticleGrid::ParticleGrid(PoissonSampler *samp)
    : sampler(samp), nodes(),
      gridMasses(), gridVelocities(), gridMomentums(), gridForces()
{
    gridMasses.reserve(samp->numPoints);
    gridVelocities.reserve(samp->numPoints);
    gridMomentums.reserve(samp->numPoints);
    gridForces.reserve(samp->numPoints);
}

// helper func - Dp matrix
glm::mat3 calcDp(glm::vec3 xp, glm::vec3 xi) {
    float dx = glm::distance(xp, xi);
    float dx2 = pow(dx, 2);
    // cubic interp
    glm::mat3 Dp = (1.f/3.f) * dx2 * glm::mat3(1.f);

    // quadratic interp
//    glm::mat3 Dp = (0.25f) * dx2 * glm::mat3(1.f);

    return glm::inverse(Dp);
}

void ParticleGrid::p2gTransfer() {
    for (int i = 0; i < sampler->numPoints; i++) {
        Particle* p = sampler->finalSamples[i];
        // weight (wip)
        float wip = computeWeight(p->pos, p->gridLoc);
        glm::mat3 Dp = calcDp(p->pos, p->gridLoc);

        // updates
        gridMasses[i] = wip * p->mass;
//        float x = glm::dot((p->Bp * Dp), (p->gridLoc - p->pos));
        gridMomentums[i] = wip * p->mass * (p->vp + (p->Bp * Dp * (p->gridLoc - p->pos)));

        if (p->mass <= 0.f) {
            gridMasses[i] = 0.f;
            gridVelocities[i] = glm::vec3(0.f);
        }
        else {
            gridVelocities[i] = gridMomentums[i] / gridMasses[i];
        }
    }
}

void ParticleGrid::applyForces() {
    for (int i = 0; i < sampler->numPoints; i++) {
        Particle* p = sampler->finalSamples[i];
        glm::vec3 dwip = computeWeightGrad(p->pos, p->gridLoc);

        glm::mat3 oldDG = p->deformationGrad();
        // force is value or vector?
        gridForces[i] = -1.f * p->vol * p->stress * glm::transpose(oldDG) * dwip;

        // velocity update
        float dt = 1.f;
        gridVelocities[i] = gridVelocities[i] + dt * (gridForces[i] / gridMasses[i]);

        // update deformation gradient
//        p->deform = (glm::mat3(1.f) + dt * glm::dot(gridVelocities[i], dwip)) * p->deform;
        glm::mat3 newDG = oldDG + oldDG * (dt * glm::dot(gridVelocities[i], dwip));
        p->updateDeformationGrad(newDG);
    }
}

void ParticleGrid::g2pTransfer() {
    for (int i = 0; i < sampler->numPoints; i++) {
        Particle* p = sampler->finalSamples[i];
        // weight (wip)
        float wip = computeWeight(p->pos, p->gridLoc);

        p->vp = wip * gridVelocities[i];
        p->Bp = wip * gridVelocities[i] * (p->gridLoc - p->pos);

        // pos and force transfer?
    }
}

void ParticleGrid::particleAdvection() {
    for (int i = 0; i < sampler->numPoints; i++) {
        Particle* p = sampler->finalSamples[i];

        float dt = 1.f;
        p->pos = p->pos + dt * gridVelocities[i];
    }
}

// helper func - N(x)
float kernelFunc(float x) {
    // cubic kernel
    float val = abs(x);
    float ret = 0.f;
    if (val < 1.f) {
        ret = ((0.5f) * pow(val, 3)) - pow(val, 2) + (2.f/3.f);
    }
    else if (val < 2.f) {
        ret = (1.f/6.f) * pow((2 - val), 3);
    }
    else {
        ret = 0.f;
    }

    // quadratic kernel
//    if (val < 0.5f) {
//        ret = (3/4) - pow(val, 2);
//    }
//    else if (val < 1.5f) {
//        ret = 0.5f * pow((1.5f - val), 2)
//    }
//    else {
//        ret = 0.f;
//    }

    // linear kernel
//    if (val < 1.f) {
//        ret = 1.f - val;
//    }
//    else {
//        ret = 0.f;
//    }

    return ret;
}

// helper func - N'(x)
float kernelDerivFunc(float x) {
    // cubic kernel
    float val = abs(x);
    float ret = 0.f;
    if (val < 1.f) {
        ret = (1.5f * pow(val, 2)) - (2.f * val);
    }
    else if (val < 2.f) {
        ret = -0.5f * pow((2.f - val), 2);
    }
    else {
        ret = 0.f;
    }

    // quadratic kernel
//    if (val < 0.5f) {
//        ret = -0.5f * val;
//    }
//    else if (val < 1.5f) {
//        ret = -1.5f + val;
//    }
//    else {
//        ret = 0.f;
//    }

    // linear kernel
//    if (val < 1.f) {
//        ret = -1.f
//    }
//    else {
//        ret = 0.f;
//    }

    return ret;
}

float ParticleGrid::computeWeight(glm::vec3 evalPos, glm::vec3 gridPos) {
    // interpolation func:
    // wip = N((1/h)(xp - xi)) N((1/h)(yp - yi)) N((1/h)(zp - zi))
    // xp = evalpos, xi = gridpos
    float hi = 1.f / sampler->cellSize;
    float xKern = kernelFunc(hi * (evalPos.x - gridPos.x));  // a
    float yKern = kernelFunc(hi * (evalPos.y - gridPos.y));  // b
    float zKern = kernelFunc(hi * (evalPos.z - gridPos.z));  // c

    return xKern * yKern * zKern;
}

// idk whether to return mat or vec??
glm::vec3 ParticleGrid::computeWeightGrad(glm::vec3 evalPos, glm::vec3 gridPos) {
    float hi = 1.f / sampler->cellSize;
    float xKern = kernelFunc(hi * (evalPos.x - gridPos.x));  // a
    float yKern = kernelFunc(hi * (evalPos.y - gridPos.y));  // b
    float zKern = kernelFunc(hi * (evalPos.z - gridPos.z));  // c
    float xKernDeriv = hi * kernelDerivFunc(hi * (evalPos.x - gridPos.x));
    float yKernDeriv = hi * kernelDerivFunc(hi * (evalPos.y - gridPos.y));
    float zKernDeriv = hi * kernelDerivFunc(hi * (evalPos.z - gridPos.z));

//    glm::vec3 x = glm::vec3(xKernDeriv, yKern, zKern);
//    glm::vec3 y = glm::vec3(xKern, yKernDeriv, zKern);
//    glm::vec3 z = glm::vec3(xKern, yKern, zKernDeriv);

//    glm::mat3 ret = glm::mat3(x, y, z);
    glm::vec3 ret;
    ret.x = xKernDeriv * yKern * zKern;
    ret.y = xKern * yKernDeriv * zKern;
    ret.z = xKern * yKern * zKernDeriv;

    return ret;
}
