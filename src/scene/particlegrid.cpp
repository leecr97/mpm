#include "particlegrid.h"


ParticleGrid::ParticleGrid(PoissonSampler *sampler)
    : sampler(sampler)
{
    initialize();
}

void ParticleGrid::initialize() {
    gridDim[0] = sampler->gridDim[0] + 10;
    gridDim[1] = sampler->gridDim[1] + 10;
    gridDim[2] = sampler->gridDim[2] + 10;
    cellSize = sampler->cellSize;
    gridMasses = std::vector<std::vector<std::vector<float>>>(
                  gridDim[0],
                  std::vector<std::vector<float>>(
                  gridDim[1],
                  std::vector<float>(
                  gridDim[2], 0.f)));
    gridVelocities = std::vector<std::vector<std::vector<glm::vec3>>>(
                  gridDim[0],
                  std::vector<std::vector<glm::vec3>>(
                  gridDim[1],
                  std::vector<glm::vec3>(
                  gridDim[2], glm::vec3(0.0f))));
    gridMomentums = std::vector<std::vector<std::vector<glm::vec3>>>(
                  gridDim[0],
                  std::vector<std::vector<glm::vec3>>(
                  gridDim[1],
                  std::vector<glm::vec3>(
                  gridDim[2], glm::vec3(0.0f))));
    gridForces = std::vector<std::vector<std::vector<glm::vec3>>>(
                  gridDim[0],
                  std::vector<std::vector<glm::vec3>>(
                  gridDim[1],
                  std::vector<glm::vec3>(
                  gridDim[2], glm::vec3(0.0f))));
}

void ParticleGrid::reset() {
    for (int x = 0; x < gridDim[0]; x++) {
        for (int y = 0; y < gridDim[1]; y++) {
            for (int z = 0; z < gridDim[2]; z++) {
                gridMasses[x][y][z] = 0.f;
                gridVelocities[x][y][z] = glm::vec3(0.0f);
                gridMomentums[x][y][z] = glm::vec3(0.0f);
                gridForces[x][y][z] = glm::vec3(0.0f);
            }
        }
    }
}

void ParticleGrid::MPM() {
    reset();

    // particle to grid
    p2gTransfer();
    computeGridVelocities();

    // updates - force, velocity, DG
    computeForces();
    velocityUpdate();
    updateDGs();

    // grid to particle
    g2pTransfer();
    particleAdvection();
}

// helper func - Dp matrix
glm::mat3 calcDp(glm::vec3 xp, glm::vec3 xi) {
    float dx = glm::distance(xp, xi);
    float dx2 = pow(dx, 2);
    // cubic interp
//    glm::mat3 Dp = (1.f/3.f) * dx2 * glm::mat3(1.f);

    // quadratic interp
    glm::mat3 Dp = (0.25f) * dx2 * glm::mat3(1.f);

    return glm::inverse(Dp);
}

void ParticleGrid::computeWeights() {
    for (int i = 0; i < sampler->finalSamples.size(); i++) {
        Particle* p = &(sampler->finalSamples[i]);
        glm::vec3 loc = posToGrid(p->pos);

        p->weight = weightFunc(p->pos, loc);
        p->weightGrad = weightGradFunc(p->pos, loc);
    }
}

void ParticleGrid::p2gTransfer() {
    // pre pass - compute particle weights
    computeWeights();

//    std::cout << "p2g ";
    for (int i = 0; i < sampler->finalSamples.size(); i++) {
        Particle* p = &(sampler->finalSamples[i]);
        glm::vec3 loc = posToGrid(p->pos);
        glm::vec3 gridPos = gridToPos(loc.x, loc.y, loc.z);

        // weight (wip)
        float wip = p->weight;
        glm::mat3 Dp = calcDp(p->pos, gridPos);

        // updates - mass and momentum
        float currMass = wip * p->mass;
        gridMasses[loc[0]][loc[1]][loc[2]] += currMass;

        glm::vec3 currMom = wip * p->mass * (p->vp + (p->Bp * Dp * (gridPos - p->pos)));
        gridMomentums[loc[0]][loc[1]][loc[2]] += currMom;
    }
}

void ParticleGrid::computeGridVelocities() {
//    std::cout << "velocities ";
    for (int i = 0; i < sampler->finalSamples.size(); i++) {
        Particle* p = &(sampler->finalSamples[i]);
        glm::vec3 loc = posToGrid(p->pos);

        if (gridMasses[loc[0]][loc[1]][loc[2]] <= 0.f) {
            gridMasses[loc[0]][loc[1]][loc[2]] = 0.f;
            gridVelocities[loc[0]][loc[1]][loc[2]] = glm::vec3(0.f);
        }
        else {
            glm::vec3 currVel = gridMomentums[loc[0]][loc[1]][loc[2]] / gridMasses[loc[0]][loc[1]][loc[2]];
            gridVelocities[loc[0]][loc[1]][loc[2]] = currVel;
        }
    }
}

void ParticleGrid::computeForces() {
//    std::cout << "forces ";
    for (int i = 0; i < sampler->finalSamples.size(); i++) {
        Particle* p = &(sampler->finalSamples[i]);
        glm::vec3 loc = posToGrid(p->pos);
        glm::vec3 dwip = p->weightGrad;

        glm::mat3 DG = p->deformationGrad();
        // force is value or vector?
        glm::vec3 currForce = -1.f * p->vol * p->stress * glm::transpose(DG) * dwip;
        gridForces[loc[0]][loc[1]][loc[2]] += currForce;
    }
}

void ParticleGrid::velocityUpdate() {
//    std::cout << "updateVels ";
    for (int i = 0; i < sampler->finalSamples.size(); i++) {
        Particle* p = &(sampler->finalSamples[i]);
        glm::vec3 loc = posToGrid(p->pos);

        if (gridMasses[loc[0]][loc[1]][loc[2]] != 0) {
            gridVelocities[loc[0]][loc[1]][loc[2]] +=
                    dt * (gridForces[loc[0]][loc[1]][loc[2]] / gridMasses[loc[0]][loc[1]][loc[2]]);
        }
        gridVelocities[loc[0]][loc[1]][loc[2]] += glm::vec3(0, -2.0f, 0);

        if (collisionCheck(loc)) {
            gridVelocities[loc[0]][loc[1]][loc[2]] = glm::vec3(0.0f);
        }
    }
}

bool ParticleGrid::collisionCheck(glm::vec3 loc) {
    if (loc[0] < 2 || loc[0] >= gridDim[0] - 2 ||
        loc[1] < 2 || loc[1] >= gridDim[1] - 2 ||
        loc[2] < 2 || loc[2] >= gridDim[2] - 2) {
        return true;
    }

    return false;
}

void ParticleGrid::updateDGs() {
//    std::cout << "updateDG ";
    for (int i = 0; i < sampler->finalSamples.size(); i++) {
        Particle* p = &(sampler->finalSamples[i]);
        glm::vec3 loc = posToGrid(p->pos);
        glm::vec3 dwip = p->weightGrad;

        // update deformation gradient
        glm::mat3 oldDG = p->deformationGrad();

        glm::vec3 velSum = glm::vec3(0.0f);
        for (int x = loc[0] - 2; x <= loc[0] + 1; x++) {
            for (int y = loc[1] - 2; y <= loc[1] + 1; y++) {
                for (int z = loc[2] - 2; z <= loc[2] + 1; z++) {
                    glm::vec3 currVel = gridVelocities[x][y][z];
                    velSum += currVel;
                }
            }
        }

        glm::mat3 newDG = oldDG + oldDG * (dt * glm::outerProduct(velSum, dwip));
        p->update(newDG);
    }
}

void ParticleGrid::g2pTransfer() {
//    std::cout << "g2p ";
    for (int i = 0; i < sampler->finalSamples.size(); i++) {
        Particle* p = &(sampler->finalSamples[i]);
        glm::vec3 loc = posToGrid(p->pos);
        // weight (wip)
        float wip = p->weight;

        // update particle velocity and Bp
        glm::vec3 newVel = glm::vec3(0.0f);
        glm::mat3 newAff = glm::mat3(0.0f);
        for (int x = loc[0] - 2; x <= loc[0] + 1; x++) {
            for (int y = loc[1] - 2; y <= loc[1] + 1; y++) {
                 for (int z = loc[2] - 2; z <= loc[2] + 1; z++) {
                     newVel += wip * gridVelocities[x][y][z];

                     glm::vec3 gridPos = gridToPos(x,y,z);
                     glm::vec3 dist = gridPos - p->pos;
                     newAff += wip * glm::outerProduct(gridVelocities[x][y][z], dist);
                 }
            }
        }

        p->vp = newVel;
        p->Bp = newAff;
    }
}

void ParticleGrid::particleAdvection() {
//    std::cout << "particleAdvect " << std::endl;
    for (int i = 0; i < sampler->finalSamples.size(); i++) {
        Particle* p = &(sampler->finalSamples[i]);
        glm::vec3 newPos = p->pos + dt * p->vp;

        // making an obstacle
        if (newPos.x > -1.f && newPos.x < 1.f && newPos.y < -4.0f) {
            newPos.y = -4.0f;
        }

        // clamping to walls
        float ground = sampler->bounds->min[1] - 1.5f;
        if (newPos.y < ground) {
            newPos.y = ground;
            p->vp = glm::vec3(0.0f);
        }
        float ceil = sampler->bounds->max[1] + 1.5f;
        if (newPos.y > ceil) {
            newPos.y = ceil;
            p->vp = glm::vec3(0.0f);
        }
        float left = sampler->bounds->min[0] - 1.5f;
        if (newPos.x < left) {
            newPos.x = left;
            p->vp = glm::vec3(0.0f);
        }
        float right = sampler->bounds->max[0] + 1.5f;
        if (newPos.x > right) {
            newPos.x = right;
            p->vp = glm::vec3(0.0f);
        }
        float front = sampler->bounds->min[2] - 1.5f;
        if (newPos.z < front) {
            newPos.z = front;
            p->vp = glm::vec3(0.0f);
        }
        float back = sampler->bounds->max[2] + 1.5f;
        if (newPos.z > back) {
            newPos.z = back;
            p->vp = glm::vec3(0.0f);
        }

        p->pos = newPos;

    }
}


// helper func - N(x)
float kernelFunc(float x) {
    float val = abs(x);
    float ret = 0.f;
    // cubic kernel
//    if (val < 1.f) {
//        ret = ((0.5f) * pow(val, 3)) - pow(val, 2) + (2.f/3.f);
//    }
//    else if (val < 2.f) {
//        ret = (1.f/6.f) * pow((2 - val), 3);
//    }
//    else {
//        ret = 0.f;
//    }

    // quadratic kernel
    if (val < 0.5f) {
        ret = 0.75f - pow(val, 2);
    }
    else if (val < 1.5f) {
        ret = 0.5f * pow((1.5f - val), 2);
    }
    else {
        ret = 0.f;
    }

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
    float val = abs(x);
    float ret = 0.f;
    // cubic kernel
//    if (val < 1.f) {
//        ret = (1.5f * pow(val, 2)) - (2.f * val);
//    }
//    else if (val < 2.f) {
//        ret = -0.5f * pow((2.f - val), 2);
//    }
//    else {
//        ret = 0.f;
//    }

    // quadratic kernel
    if (val < 0.5f) {
        ret = -2.f * val;
    }
    else if (val < 1.5f) {
        ret = -1.5f + val;
    }
    else {
        ret = 0.f;
    }

    // linear kernel
//    if (val < 1.f) {
//        ret = -1.f
//    }
//    else {
//        ret = 0.f;
//    }

    return ret;
}

float ParticleGrid::weightFunc(glm::vec3 evalPos, glm::vec3 gridInd) {
    // interpolation func:
    // wip = N((1/h)(xp - xi)) N((1/h)(yp - yi)) N((1/h)(zp - zi))
    // xp = evalpos, xi = gridpos

    float hi = 1.f / cellSize;
    float ret = 0.0f;

    for (int x = gridInd[0] - 2; x <= gridInd[0] + 2; x++) {
        for (int y = gridInd[1] - 2; y <= gridInd[1] + 2; y++) {
            for (int z = gridInd[2] - 2; z <= gridInd[2] + 2; z++) {
                glm::vec3 gridPos = gridToPos(x,y,z);

                float xKern = kernelFunc(hi * (evalPos.x - gridPos.x));
                float yKern = kernelFunc(hi * (evalPos.y - gridPos.y));
                float zKern = kernelFunc(hi * (evalPos.z - gridPos.z));

                ret += (xKern * yKern * zKern);
            }
        }
    }

    return ret;
}

// idk whether to return mat or vec??
glm::vec3 ParticleGrid::weightGradFunc(glm::vec3 evalPos, glm::vec3 gridInd) {
    float hi = 1.f / cellSize;
    glm::vec3 ret = glm::vec3(0.0f);

    for (int x = gridInd[0] - 2; x <= gridInd[0] + 2; x++) {
       for (int y = gridInd[1] - 2; y <= gridInd[1] + 2; y++) {
           for (int z = gridInd[2] - 2; z <= gridInd[2] + 2; z++) {
               glm::vec3 gridPos = gridToPos(x,y,z);

               float xKern = kernelFunc(hi * (evalPos.x - gridPos.x));  // a
               float yKern = kernelFunc(hi * (evalPos.y - gridPos.y));  // b
               float zKern = kernelFunc(hi * (evalPos.z - gridPos.z));  // c
               float xKernDeriv = hi * kernelDerivFunc(hi * (evalPos.x - gridPos.x));
               float yKernDeriv = hi * kernelDerivFunc(hi * (evalPos.y - gridPos.y));
               float zKernDeriv = hi * kernelDerivFunc(hi * (evalPos.z - gridPos.z));

               ret.x += xKernDeriv * yKern * zKern;
               ret.y += xKern * yKernDeriv * zKern;
               ret.z += xKern * yKern * zKernDeriv;
           }
       }
    }

    return ret;
}

glm::vec3 ParticleGrid::posToGrid(glm::vec3 p) {
    glm::vec3 min = sampler->bounds->min;

    int x = (int)(glm::clamp(((p[0] - min[0])/cellSize), 0.f, gridDim[0] - 11));
    int y = (int)(glm::clamp(((p[1] - min[1])/cellSize), 0.f, gridDim[1] - 11));
    int z = (int)(glm::clamp(((p[2] - min[2])/cellSize), 0.f, gridDim[2] - 11));

    return glm::vec3(x + 5, y + 5, z + 5);
}

glm::vec3 ParticleGrid::gridToPos(int locx, int locy, int locz) {
    glm::vec3 min = sampler->bounds->min;

    float posx = min.x + (locx - 5) * cellSize;
    float posy = min.y + (locy - 5) * cellSize;
    float posz = min.z + (locz - 5) * cellSize;

    return glm::vec3(posx, posy, posz);
}
