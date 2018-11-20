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

    // opengl:mat3 and eigen:mat3 are column major so no need to complicate conversions
    // copy over to eigen for svd calculation
    EigenMat3 eigen_newElas = EigenMat3();
    EigenMat3 eigen_newPlas = EigenMat3();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            eigen_newElas(i, j) = newElas[i][j];
            eigen_newPlas(i, j) = plasticity[i][j];
        }
    }

    // find svd of elas
    Eigen::JacobiSVD<EigenMat3> eigen_SVD(eigen_newElas, Eigen::ComputeFullU | Eigen::ComputeFullV);
    EigenMat3 eigen_U = eigen_SVD.matrixU();
    auto eigen_S = eigen_SVD.singularValues();
    EigenMat3 eigen_V = eigen_SVD.matrixV();

    // define the small constants thetaC and thetaS for clamping
    float thetaC = 0.25f;
    float thetaS = 0.075f;

    // swap u and v columns to make diagonals appropriately for clamping and SVD
    //auto size = eigen_S.rows() * eigen_S.cols();
    auto size = 3 * 1;
    for (int i = 1; i < size; ++i) {
        for (int j = i; j > 0; --j) {
            if (eigen_S[j - 1] < eigen_S[j]) { continue; }

            // swapping
            std::swap(eigen_S[j], eigen_S[j - 1]);
            for (int k = 0; k < 3; ++k) {
                // swap row j with row j - 1
                std::swap(eigen_U(k, j), eigen_U(k, j - 1));
                std::swap(eigen_V(k, j), eigen_V(k, j - 1));
            }
        }
    }
    // if determinant is -1 flip the sign of col3 of u or v and sigma
    if (eigen_U.determinant() < 0) {
        // multiply U's last column by -1
        eigen_U.col(2) *= -1.0;
        // multiply sigma's last row first entry by -1
        eigen_S[2] *= -1.0;
    }
    if (eigen_V.determinant() < 0) {
        // multiply V's last column by -1
        eigen_V.col(2) *= -1.0;
        // multiply sigma's last row first entry by -1
        eigen_S[2] *= -1.0;
    }
    // check have valid determinants
    if (!(eigen_U.determinant() > 0 && eigen_V.determinant() > 0)) {
        std::cout<<"determinants of u and v in svd decomp "
                 <<"were invalid (non inversible matrices)."<<std::endl;
        throw;
    }
    // clamping based on 1 - thetaC and 1 + thetaS values -- done in conversion to glm
//    for (int i = 0; i < 3; i++) {
//        eigen_S[i] = std::clamp(eigen_S[i], 1 - thetaC, 1 + thetaS);
//    }
    // todo - maybe ? add in an additional check here to see if the reverse of the decomposition
    // brings back the original inputted deformation gradient

    // yogurt values?
    float mu = 0.2f;
    float lambda = 0.3f;

    EigenMat3 Fe = eigen_newElas;
    EigenMat3 Fp = eigen_newPlas;

    // calculate stress
    EigenMat3 R = eigen_U * eigen_V.transpose();
    float j_elastic = Fe.determinant();
    float j_plastic = Fp.determinant();
    auto stress_corrotated = 2.0 * mu * (Fe - R) + lambda * (j_elastic - 1.0) * Fe.inverse().transpose();
    auto stress_piola_kirchoff = 2.0 * mu * (Fe - R) + lambda * j_elastic * (j_elastic - 1.0) * Fe.inverse().transpose();

    // convert from eigen to glm (and clamp S)
    glm::mat3 realU; glm::mat3 realS; glm::mat3 realV;

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            realU[i][j] = eigen_U(i, j);

            if (eigen_S(i, j) < 1 - thetaC) {
                realS[i][j] = 1 - thetaC;
            }
            else if (eigen_S(i, j) > 1 + thetaS) {
                realS[i][j] = 1 + thetaS;
            }
            else {
                realS[i][j] = eigen_S(i, j);
            }

            realV[i][j] = eigen_V(i, j);
        }
    }

    elasticity = realU * realS * realV;

    // find plasticity from this elasticity and newDG
    plasticity = glm::inverse(elasticity) * newDG;
}
