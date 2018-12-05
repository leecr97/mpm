#include "particlegrid.h"


ParticleGrid::ParticleGrid(PoissonSampler *samp)
    : sampler(samp)
{
    gridDim[0] = samp->gridDim[0] + 4;
    gridDim[1] = samp->gridDim[1] + 4;
    gridDim[2] = samp->gridDim[2] + 4;
    cellSize = samp->cellSize;
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
    for (Particle* p : sampler->finalSamples) {
        glm::vec3 loc = posToGrid(p->pos);

        p->weight = weightFunc(p->pos, loc);
        p->weightGrad = weightGradFunc(p->pos, loc);

//        if (p->id == 100) {
//            std::cout << "weight: " << p->weight << std::endl;
//            std::cout << "wgrad: " << p->weightGrad.x << ", "
//                                   << p->weightGrad.y << ", "
//                                   << p->weightGrad.z << std::endl;
//        }
    }
}

void ParticleGrid::p2gTransfer() {
    // pre pass - compute particle weights
    computeWeights();

//    std::cout << "p2g ";
    for (Particle* p : sampler->finalSamples) {
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

//        if (p->id == 100) {
//            std::cout << "mass: " << gridMomentums[loc[0]][loc[1]][loc[2]].x << ", "
//                                  << gridMomentums[loc[0]][loc[1]][loc[2]].y << ", "
//                                  << gridMomentums[loc[0]][loc[1]][loc[2]].z << std::endl;
//            std::cout << "momentum: " << gridMomentums[loc[0]][loc[1]][loc[2]].x << ", "
//                                  << gridMomentums[loc[0]][loc[1]][loc[2]].y << ", "
//                                  << gridMomentums[loc[0]][loc[1]][loc[2]].z << std::endl;
//        }
    }
}

void ParticleGrid::computeGridVelocities() {
//    std::cout << "velocities ";
    for (Particle* p : sampler->finalSamples) {
        glm::vec3 loc = posToGrid(p->pos);

        if (gridMasses[loc[0]][loc[1]][loc[2]] <= 0.f) {
            gridMasses[loc[0]][loc[1]][loc[2]] = 0.f;
            gridVelocities[loc[0]][loc[1]][loc[2]] = glm::vec3(0.f);
        }
        else {
            glm::vec3 currVel = gridMomentums[loc[0]][loc[1]][loc[2]] / gridMasses[loc[0]][loc[1]][loc[2]];
            gridVelocities[loc[0]][loc[1]][loc[2]] = currVel;
        }

//        if (p->id == 100) {
//            std::cout << "vel: " << gridVelocities[loc[0]][loc[1]][loc[2]].x << ", "
//                                 << gridVelocities[loc[0]][loc[1]][loc[2]].y << ", "
//                                 << gridVelocities[loc[0]][loc[1]][loc[2]].z << std::endl;
//        }
    }
}

void ParticleGrid::computeForces() {
//    std::cout << "forces ";
    for (Particle* p : sampler->finalSamples) {
        glm::vec3 loc = posToGrid(p->pos);
        glm::vec3 dwip = p->weightGrad;

        glm::mat3 DG = p->deformationGrad();
        // force is value or vector?
        glm::vec3 currForce = -1.f * p->vol * p->stress * glm::transpose(DG) * dwip;
        gridForces[loc[0]][loc[1]][loc[2]] += currForce;

//        if (p->id == 100) {
//            std::cout << "force: " << gridForces[loc[0]][loc[1]][loc[2]].x << ", "
//                                 << gridForces[loc[0]][loc[1]][loc[2]].y << ", "
//                                 << gridForces[loc[0]][loc[1]][loc[2]].z << std::endl;
//        }
    }
}

void ParticleGrid::velocityUpdate() {
//    std::cout << "updateVels ";
    for (Particle* p : sampler->finalSamples) {
        glm::vec3 loc = posToGrid(p->pos);

        if (gridMasses[loc[0]][loc[1]][loc[2]] != 0) {
            gridVelocities[loc[0]][loc[1]][loc[2]] +=
                    dt * (gridForces[loc[0]][loc[1]][loc[2]] / gridMasses[loc[0]][loc[1]][loc[2]]);
        }
        gridVelocities[loc[0]][loc[1]][loc[2]] += glm::vec3(0, -2.0f, 0);

        if (collisionCheck(loc)) {
            gridVelocities[loc[0]][loc[1]][loc[2]] = glm::vec3(0.0f);
        }

//        if (p->id == 100) {
//            std::cout << "new vel: " << gridVelocities[loc[0]][loc[1]][loc[2]].x << ", "
//                                 << gridVelocities[loc[0]][loc[1]][loc[2]].y << ", "
//                                 << gridVelocities[loc[0]][loc[1]][loc[2]].z << std::endl;
//        }
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
    for (Particle* p : sampler->finalSamples) {
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

        glm::mat3 newDG = oldDG + oldDG * (dt * glm::dot(velSum, dwip));
        p->update(newDG);
    }
}

void ParticleGrid::g2pTransfer() {
//    std::cout << "g2p ";
    for (Particle* p : sampler->finalSamples) {
        glm::vec3 loc = posToGrid(p->pos);
        // weight (wip)
        float wip = p->weight;

        // update particle velocity and Bp
        p->vp = glm::vec3(0.0f);
        p->Bp = glm::vec3(0.0f);
        for (int x = loc[0] - 2; x <= loc[0] + 1; x++) {
            for (int y = loc[1] - 2; y <= loc[1] + 1; y++) {
                 for (int z = loc[2] - 2; z <= loc[2] + 1; z++) {
                     p->vp += wip * gridVelocities[x][y][z];

                     glm::vec3 gridPos = gridToPos(x,y,z);
                     p->Bp += wip * gridVelocities[x][y][z] * (gridPos - p->pos);
                 }
            }
        }

//        if (p->id == 100) {
//            std::cout << "vp: " << p->vp.x << ", "
//                                << p->vp.y << ", "
//                                << p->vp.z << std::endl;
//            std::cout << "Bp: " << p->Bp.x << ", "
//                                << p->Bp.y << ", "
//                                << p->Bp.z << std::endl;
//        }

//        std::cout << "vel: " << p->vp[0] << ", " << p->vp[1] << ", " << p->vp[2] <<std::endl;
    }
}

void ParticleGrid::particleAdvection() {
//    std::cout << "particleAdvect " << std::endl;
    for (Particle* p : sampler->finalSamples) {
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
            p->Bp = glm::vec3(0.0f);
        }
        float ceil = sampler->bounds->max[1] + 1.5f;
        if (newPos.y > ceil) {
            newPos.y = ceil;
            p->vp = glm::vec3(0.0f);
            p->Bp = glm::vec3(0.0f);
        }
        float left = sampler->bounds->min[0] - 1.5f;
        if (newPos.x < left) {
            newPos.x = left;
            p->vp = glm::vec3(0.0f);
            p->Bp = glm::vec3(0.0f);
        }
        float right = sampler->bounds->max[0] + 1.5f;
        if (newPos.x > right) {
            newPos.x = right;
            p->vp = glm::vec3(0.0f);
            p->Bp = glm::vec3(0.0f);
        }
        float front = sampler->bounds->min[2] - 1.5f;
        if (newPos.z < front) {
            newPos.z = front;
            p->vp = glm::vec3(0.0f);
            p->Bp = glm::vec3(0.0f);
        }
        float back = sampler->bounds->max[2] + 1.5f;
        if (newPos.z > back) {
            newPos.z = back;
            p->vp = glm::vec3(0.0f);
            p->Bp = glm::vec3(0.0f);
        }

        p->pos = newPos;

//        if (p->id == 100) {
//            std::cout << "pos: " << p->pos.x << ", "
//                                << p->pos.y << ", "
//                                << p->pos.z << std::endl;
//        }
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

//    std::cout << "weight: " << ret << std::endl;
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


//    glm::vec3 x = glm::vec3(xKernDeriv, yKern, zKern);
//    glm::vec3 y = glm::vec3(xKern, yKernDeriv, zKern);
//    glm::vec3 z = glm::vec3(xKern, yKern, zKernDeriv);

//    glm::mat3 ret = glm::mat3(x, y, z);

    return ret;
}

glm::vec3 ParticleGrid::posToGrid(glm::vec3 p) {
    glm::vec3 min = sampler->bounds->min;

    int x = (int)(glm::clamp(((p[0] - min[0])/cellSize), 0.f, gridDim[0] - 5));
    int y = (int)(glm::clamp(((p[1] - min[1])/cellSize), 0.f, gridDim[1] - 5));
    int z = (int)(glm::clamp(((p[2] - min[2])/cellSize), 0.f, gridDim[2] - 5));

//    std::cout << "grid: " << x << ", " << y << ", " << z << std::endl;
    return glm::vec3(x + 2, y + 2, z + 2);
}

glm::vec3 ParticleGrid::gridToPos(int locx, int locy, int locz) {
    glm::vec3 min = sampler->bounds->min;

    float posx = min.x + (locx - 2) * cellSize;
    float posy = min.y + (locy - 2) * cellSize;
    float posz = min.z + (locz - 2) * cellSize;

//    std::cout << "pos: " << posx << ", " << posy << ", " << posz << std::endl;
    return glm::vec3(posx, posy, posz);
}
