#include "poissonsampler.h"

PoissonSampler::PoissonSampler(Mesh &mesh, Scene &scene, bool isThreeDim)
    : m(mesh), s(scene), bvh(nullptr), bounds(nullptr), finalSamples(0, nullptr),
      origPositions(), threeDim(isThreeDim), gridDim(glm::vec3(0.0f)),
      K(30), numPoints(0), samp(5, 5) {

    initialize();
    generateSamples();
}

//set up background grid based on radius
void PoissonSampler::initialize() {
    int nDim = (threeDim) ? 3 : 2;

    cellSize = radius/sqrt(nDim);
//    cellSize = radius/((nDim*nDim));
    bvh = new PoissonBVH(m);
    bounds = bvh->root->bbox;

    Point3f maxP = bounds->max;
    Point3f minP = bounds->min;

    this->gridDim = glm::vec3(glm::ceil((maxP[0] - minP[0])/cellSize),
                               glm::ceil((maxP[1] - minP[1])/cellSize),
                               glm::ceil((maxP[2] - minP[2])/cellSize) );
}

GLenum PoissonSampler::drawMode() const {
    return GL_POINTS;
}

void PoissonSampler::create() {
    //create new vbo
    //set drawmode to GL_POINTS

    //  pos vec3s
    std::vector<glm::vec3> posVector;

    for (Particle* p: finalSamples ){
        posVector.push_back(p->pos);
    }

    GLuint points_idx[numPoints];
    glm::vec3 points_vert_pos[numPoints];
    glm::vec3 points_vert_nor[numPoints];
    glm::vec3 points_vert_col[numPoints];

    glm::vec3 color2 = glm::vec3(255.0f, 255.0f, 255.0f) / 255.0f;
    for (int i = 0; numPoints!=0 && i<numPoints; i++) {
        points_vert_col[i] = (color2);
        points_vert_nor[i] = glm::vec3(0, 0, 1);
        points_idx[i] = i;
        points_vert_pos[i] = posVector[i];
    }

    count = numPoints;

    //  handling line indices vbo
    bufIdx.create();
    bufIdx.bind();
    bufIdx.setUsagePattern(QOpenGLBuffer::StaticDraw);
    bufIdx.allocate(points_idx, numPoints * sizeof(GLuint));

    bufPos.create();
    bufPos.bind();
    bufPos.setUsagePattern(QOpenGLBuffer::StaticDraw);
    bufPos.allocate(points_vert_pos, numPoints * sizeof(glm::vec3));

    bufNor.create();
    bufNor.bind();
    bufNor.setUsagePattern(QOpenGLBuffer::StaticDraw);
    bufNor.allocate(points_vert_nor, numPoints * sizeof(glm::vec3));

    bufCol.create();
    bufCol.bind();
    bufCol.setUsagePattern(QOpenGLBuffer::StaticDraw);
    bufCol.allocate(points_vert_col, numPoints * sizeof(glm::vec3));
}

void PoissonSampler::fallWithGravity() {
    for (Particle* p : finalSamples) {
        if (p->pos.y >= -2) {
//            std::cout << "y: " << p->pos.y << std::endl;
            p->pos.y = p->pos.y - 0.2f;
            // force of gravity * dt
//            -2.f * p->mp;
            create();
        }
    }
}

void PoissonSampler::resetParticlePositions() {
    if (finalSamples.size() != origPositions.size()) {
        std::cout << "something went wrong" << std::endl;
    }
    for (int i = 0; i < finalSamples.size(); i++) {
        Particle* p = finalSamples[i];
        p->pos = origPositions[i];
    }
    create();
}

void PoissonSampler::generateSamples(){
    std::cout << "generating samples..." << std::endl;
    // active sample list, final sample list
    std::vector<Particle*> activeSamples(0, nullptr);
    std::vector<Particle*> finSamples(0, nullptr);

    // background grid
    std::vector<std::vector<std::vector<Particle*>>> backgroundGrid
                = std::vector<std::vector<std::vector<Particle*>>>(
                  gridDim[0],
                  std::vector<std::vector<Particle*>>(
                  gridDim[1],
                  std::vector<Particle*>(
                  gridDim[2], nullptr)));

    // choose a random voxel of grid, make initial sample
    int randX = rand() % static_cast<int>(gridDim[0]);  // rand() % gridDim[0]
    int randY = rand() % static_cast<int>(gridDim[1]);
    int randZ = rand() % static_cast<int>(gridDim[2]);

    glm::vec3 initial = glm::vec3(randX, randY, randZ);
    Particle* start = new Particle(initial, m.transform.position(), 0);

    // insert into background grid and active list
    backgroundGrid[initial[0]][initial[1]][(threeDim) ? initial[2] : 0] = start;
    activeSamples.push_back(start);

    // active list loop
    while(activeSamples.size() > 0) {
        // choose random point
        int ra = rand() % activeSamples.size();
        Particle* xi = activeSamples[ra];
        bool foundNewPoint = false;

        // generate up to k points chosen uniformly from r to 2r
        for (int i=0; i<K; i++) {
            glm::vec3 pos = randomLocAround(xi->pos);
            glm::vec3 gLoc = posToGridLoc(pos);

            // check if point is within distance r of existing samples
            // aka check the 8 boxes around current gridLoc
            glm::vec3 div = glm::vec3(gridDim)/radius;
            glm::vec3 MIN(0.0f);
            glm::vec3 MAX(0.0f);
            for (int j=0; j<3; j++) {
                MIN[j] = pos[j] - 4*div[j];
                MAX[j] = pos[j] + 4*div[j];
            }
            MIN = posToGridLoc(MIN);
            MAX = posToGridLoc(MAX);

            bool valid = true;
            bool withinR = false;
            for (int x = MIN[0]; x <= MAX[0]; x++) {
                for (int y = MIN[1]; y <= MAX[1]; y++) {
                    for(int z = MIN[2]; z <= MAX[2]; z++) {
                        if (backgroundGrid[x][y][z] != nullptr) {
                            // not valid, same loc as xi
                            if (x == gLoc[0] && y == gLoc[1] && z == gLoc[2]) {
                                valid = false;
                            }

                            // distance check
                            if (glm::distance(pos, xi->pos) >= radius && (glm::distance(pos, xi->pos) <= 2*radius)) {
                                withinR = true;
                            }

                            valid &= withinR;
                        }
                    }
                }
            }

            // if adequately far, add to active list
            if (valid && validWithinBounds(pos)) {
                Particle* kPoint = new Particle(gLoc, pos, activeSamples.size() + i);

                activeSamples.push_back(kPoint);
                backgroundGrid[gLoc[0]][gLoc[1]][gLoc[2]] = kPoint;

                foundNewPoint = true;
            }

        }

        if (!foundNewPoint) {
            // if after k attempts no point is found, remove i from active list and add to final list
            finSamples.push_back(new Particle(xi));
            numPoints += 1;
            activeSamples.erase(std::remove(activeSamples.begin(), activeSamples.end(), xi),
                                     activeSamples.end());

        }

    }

    for (Particle* s : finSamples) {
//        if (validWithinObj(s->pos)) {
            finalSamples.push_back(s);
            origPositions.push_back(s->pos);
//        }
    }
    std::cout << "done generating samples!" << std::endl;
}

glm::vec3 PoissonSampler::randomLocAround(glm::vec3 pos) {
    float val = 3.0f;
    float x = samp.Get2D().x * val * radius;
    float y = samp.Get2D().x * val * radius;
    float z = samp.Get2D().x * val * radius;

    float factor = radius * val/2.0f;
    x = (x > factor) ? pos[0] + x : pos[0] - x;
    y = (y > factor) ? pos[1] + y : pos[1] - y;
    z = (z > factor) ? pos[2] + z : pos[2] - z;

    return glm::vec3(x, y, z);
}

glm::vec3 PoissonSampler::posToGridLoc(glm::vec3 p) {
    glm::vec3 min = bounds->min;

    int x = (int)(glm::clamp(((p[0] - min[0])/radius), 0.0f, gridDim[0] - 1));
    int y = (int)(glm::clamp(((p[1] - min[1])/radius), 0.0f, gridDim[1] - 1));
    int z = (threeDim) ? (int)(glm::clamp(((p[2] - min[2])/radius), 0.0f, gridDim[2] - 1)) : 0;

    return glm::vec3(x, y, z);
}

bool PoissonSampler::validWithinBounds(glm::vec3 p) {
        return (p.x > bounds->min[0] && p.y > bounds->min[1] && p.z > bounds->min[2]
                && p.x < bounds->max[0] && p.y < bounds->max[1] && p.z < bounds->max[2]);
}

void Particle::updateDeformationGrad(glm::mat3 newDG) {
    // assume all new deformation was elastic
    glm::mat3 newElas = newDG * glm::inverse(plasticity);

}
