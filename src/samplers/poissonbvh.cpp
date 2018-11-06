#include "poissonsampler.h"

PoissonBVH::PoissonBVH(Mesh& m) : root(new P_BVHNode()) {
    buildBVH(m);
}

/**
 * @brief PoissonBVH::buildBVH
 * @param m - mesh from which the BVH will be built
 */
void PoissonBVH::buildBVH(Mesh& m){
    int minNumOfTris = 20;

    // building the whole tree
    root->buildSelfAsChild(m.faces, minNumOfTris);
}

/**
 * @brief P_BVHNode::buildSelfAsChild
 * @param t - the list of triangles to possibly be added to this node
 * @param minNum - min num of tris allowed for this node
 */
void P_BVHNode::buildSelfAsChild(QList<std::shared_ptr<Triangle>>& t, int minNum) {

    // notes:
    // if current object has fewer than minNum objects then this is a leaf node
    // do the split along the longest axis
    // use surface area heuristic to determine which should be in left and which should be in right
    // build left and right children
    // done

    this->bbox = this->buildBoundingBox(t);

    // base case
    if (t.length() < minNum) {
        // leaf node
        this->tris = t;

        // checking for construction errors
        if (this->bbox == nullptr) { std::cout<<"error in P_BVHNode buildSelfASChild"<<std::endl; throw; }
        if (this->l != nullptr) { std::cout<<"error in P_BVHNode buildSelfASChild"<<std::endl; throw; }
        if (this->r != nullptr) { std::cout<<"error in P_BVHNode buildSelfASChild"<<std::endl; throw; }

        return;
    }

    // finding longest axis
    glm::vec3 diag = this->bbox->Diagonal();
    float longest = glm::max(diag[0], glm::max(diag[1], diag[2]));
    int longestAxis = (longest == diag[0]) ? 0 : (longest == diag[1] ? 1 : 2);

    QList<std::shared_ptr<Triangle>>* leftList = new QList<std::shared_ptr<Triangle>>();
    QList<std::shared_ptr<Triangle>>* rightList = new QList<std::shared_ptr<Triangle>>();
    Bounds3f* leftBox = new Bounds3f();
    Bounds3f* rightBox = new Bounds3f();

    float SA = bbox->SurfaceArea();
    splitTheTris(longestAxis, t, leftList, rightList, bbox->min, bbox->max, leftBox, rightBox, SA);

    if (leftList->length() == 0 || rightList->length() == 0) { std::cout<<"error in P_BVHNode buildSelfASChild, lists not filled in correctly in splitTheTris"<<std::endl; throw; }

    this->l = new P_BVHNode();
    this->r = new P_BVHNode();

    this->l->buildSelfAsChild(*leftList, minNum);
    this->r->buildSelfAsChild(*rightList, minNum);
}

/**
 * @brief P_BVHNode::buildBoundingBox
 * @param t - list of triangles this bbox should encompass
 * @return pointer to bounding box encompassing all triangles in the inputted list
 */
Bounds3f* P_BVHNode::buildBoundingBox(QList<std::shared_ptr<Triangle>>& t) {
    Bounds3f* b = nullptr;
    for (std::shared_ptr<Triangle> tri: t) {
        Point3f minOfTri(0.0f);
        Point3f maxOfTri(0.0f);

        for (int i=0; i<3; i++) {
            minOfTri[i] = glm::min(tri->points[0][i], glm::min(tri->points[1][i], tri->points[2][i]));
            maxOfTri[i] = glm::max(tri->points[0][i], glm::max(tri->points[1][i], tri->points[2][i]));
        }

        if (b == nullptr) {
            b = new Bounds3f(minOfTri, maxOfTri);
        } else {
            *b = Union(*b, Union(minOfTri, maxOfTri));
        }
    }
    return b;
}

/**
 * @brief P_BVHNode::splitTheTris - split based on Surface Area Heuristic and using centroids to check the triangles bins/locations
 * @param axis - longest axis of bbox over which the triangles will be sorted/split
 * @param t - list of triangles to be split
 * @param left - the QList of triangles on one side of the discovered best split location
 * @param right - the QList of triangles on the other side of the discovered best split location
 * @param leftBox - bounding box for left [filled in this method]
 * @param rightBox - bounding box for right [filled in this method]
 */
void P_BVHNode::splitTheTris(int axis, QList<std::shared_ptr<Triangle>> &t,
                             QList<std::shared_ptr<Triangle>>* &left, QList<std::shared_ptr<Triangle>>* &right,
                             Point3f min, Point3f max,
                             Bounds3f* leftBox, Bounds3f* rightBox,
                             float outerSA) {

    // notes: doing binning on the longest axis based on centroid location of triangle and sah

    // check bbox min and bbox max -> for initialization if actually created properly
    for (int i=0; i<3; i++) {
        if (bbox->min[i] > bbox->max[i])
            { std::cout<<"error in P_BVHNode splitTheTris regarding bbox being created improperly for axis:"<<i<<std::endl; throw; }
    }

    // setting up vars for all looping iterations
    float stepSize = glm::vec3((max-min)/(50.0f))[axis];
    float loc = min[axis] + stepSize;
    float minSAH = std::numeric_limits<float>::max();

    bool neverSplit = true;

    // looping through vars at diff test locations
    while (loc < max[axis]) {

        // create this iter's variables
        QList<std::shared_ptr<Triangle>>* temp_tris_left = new QList<std::shared_ptr<Triangle>>();
        QList<std::shared_ptr<Triangle>>* temp_tris_right = new QList<std::shared_ptr<Triangle>>();
        Bounds3f* temp_box_left = new Bounds3f();
        Bounds3f* temp_box_right = new Bounds3f();
        float rightCount = 0.0f;
        float leftCount = 0.0f;

        // sort across curr loc
        for (std::shared_ptr<Triangle> tri: t) {
            Point3f tri_points[3];
            tri_points[0] = tri.get()->points[0];
            tri_points[1] = tri.get()->points[1];
            tri_points[2] = tri.get()->points[2];
            Point3f centroid = (tri_points[0] + tri_points[1] + tri_points[2])/3.0f;
            if (centroid[axis] < loc) {
                leftCount += 1.0f;
                temp_tris_left->append(tri);
                temp_box_left = Union(temp_box_left, centroid);
            } else {
                rightCount += 1.0f;
                temp_tris_right->append(tri);
                temp_box_right = Union(temp_box_right, centroid);
            }
        }

        // test if curr split is good or not
        float rightSA = temp_box_right->SurfaceArea();
        float leftSA = temp_box_left->SurfaceArea();
        float SAdiff = glm::abs(rightSA * rightCount + leftSA * leftCount) / outerSA;

        if (SAdiff < minSAH) {
            neverSplit = false;

            minSAH = SAdiff;
            left->clear();
            right->clear();
            *left = *temp_tris_left;
            *right = *temp_tris_right;
            *leftBox = *temp_box_left;
            *rightBox = *temp_box_right;
        }

       // iterate to next checking location
        loc += stepSize;

    } //end: while (loc < max[axis]);

    if (neverSplit) {
        std::cout<<std::endl;
    }


    for (int i=0; i<3; i++) {
        if (leftBox->min[i] > leftBox->max[i])
            { std::cout<<"error in P_BVHNode splitTheTris leftBox never filled in for axis:"<<i<<std::endl; throw; }
        if (rightBox->min[i] > rightBox->max[i])
            { std::cout<<"error in P_BVHNode splitTheTris rightBox never filled in for axis:"<<i<<std::endl; throw; }
    }

    // done.
}

/**
 * @brief PoissonBVH::intersect - traces down BVH tree to find node of intersection if it exists
 * @param ray - ray send from any direction towards object - finding intersection from here
 * @param isect - variable to filled in with intersect location if it exists; false, otherwise.
 * @return true if intersects any nodes in the BVH; false, otherwise.
 */
bool PoissonBVH::intersect(Ray &ray, Intersection *isect, int* numIntersections, glm::mat4 viewProj) {
    return root->intersect(ray, isect, numIntersections, viewProj);
}

/**
 * @brief P_BVHNode::intersect
 * @param ray - ray send from any direction towards object - finding intersection from here
 * @param isect - variable to filled in with intersect location if it exists; false, otherwise.
 * @return true if intersects any nodes in the BVH; false, otherwise.
 */
bool P_BVHNode::intersect(Ray &ray, Intersection *isect, int* numIntersections, glm::mat4 viewProj) {

//    if (this->tris.length() <= 0) { return false; }

    if (l == nullptr && r == nullptr) {
        // hit leaf node

        bool hit = false;
        for (std::shared_ptr<Triangle> triangle: tris) {
            // return closest intersection

            Triangle* tr = triangle.get();
            Intersection* isx_test = new Intersection();

            bool intersects = tr->Intersect_PBVH(ray, isx_test, viewProj);

            if (intersects) {
                *numIntersections += 1;

                isect = (isect == nullptr) ? isx_test : ( (isx_test->t < isect->t) ? isx_test : isect );
                hit = true;
            }
        }
        return hit;

    } else if ((l == nullptr && r != nullptr) || (l!= nullptr && r == nullptr))
        { std::cout<<"error in P_BVHNode intersect childNodes not proper [ie have exactly one child node]"<<std::endl; throw; }

    float* t_intersectL = new float();
    float* t_intersectR = new float();
    bool l_hasIsx = l->bbox->Intersect(ray, t_intersectL);
    bool r_hasIsx = r->bbox->Intersect(ray, t_intersectR);

    // checking none or single case of box intersection
    if (l_hasIsx && !r_hasIsx) {
        return l->intersect(ray, isect, numIntersections, viewProj);
    } else if (!l_hasIsx && r_hasIsx) {
        return r->intersect(ray, isect, numIntersections, viewProj);
    } else if (!l_hasIsx && !r_hasIsx) {
        // did not intersect with either of the bboxes
        return false;
    }

    // else: both have box intersections intersections so now checking actual tri intersections

    // temps for tri isx
    Intersection* isx_testL = new Intersection();
    Intersection* isx_testR = new Intersection();

    l_hasIsx = l->intersect(ray, isx_testL, numIntersections, viewProj);
    r_hasIsx = r->intersect(ray, isx_testR, numIntersections, viewProj);

    // checking for recursive bbox-to-triangle intersections
    if (l_hasIsx && !r_hasIsx) {
        isect = isx_testL;
        return true;
    } else if (!l_hasIsx && r_hasIsx) {
        isect = isx_testR;
        return true;
    } else if (!l_hasIsx && !r_hasIsx) {
        return false;
    }

    // else: both recursive bbox-to-triangle intersections returned true for the test intersections;
    *isect = (isx_testL->t > isx_testR->t) ? *isx_testR : *isx_testL;
    return true;
}

