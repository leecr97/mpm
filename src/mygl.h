#ifndef MYGL_H
#define MYGL_H

#include <QOpenGLVertexArrayObject>
#include <QOpenGLShaderProgram>
#include <QRubberBand>
#include <QMouseEvent>

#include <openGL/glwidget277.h>
#include <la.h>
#include <openGL/shaderprogram.h>
#include <scene/camera.h>
#include <scene/scene.h>

#include <QTimer>
#include <QTime>
#include <QSound>

#include "samplers/poissonsampler.h"
#include "scene/particlegrid.h"

#include <QStringRef>
#include <QFile>

#include <QOpenGLVertexArrayObject>
#include <QOpenGLShaderProgram>

class PoissonSampler;
class Particle;
class ParticleGrid;

class MyGL
    : public GLWidget277
{
    Q_OBJECT
private:
    ShaderProgram prog_lambert;
    ShaderProgram prog_flat;

    // vertex attribute obj for our shaders
    QOpenGLVertexArrayObject vao;

    Camera gl_camera;

    /// Timer linked to timerUpdate(). Fires approx. 60 times per second
    QTimer timer;
    int timeCount;

    Scene scene;

    QString output_filepath;

    QSound completeSFX;
    bool gravity;

    PoissonSampler* poissonSampler;
    Mesh* poissonMesh;
    std::vector<Particle*> particles;
    ParticleGrid* pgrid;

public:
    explicit MyGL(QWidget *parent = 0);
    ~MyGL();

    void initializeGL();
    void resizeGL(int w, int h);
    void paintGL();
    void GLDrawScene();

    void loadObj();
    void poissonSamples();
    void gridForces();

protected:
    void keyPressEvent(QKeyEvent *e);

private slots:
    /// Slot that gets called ~60 times per second
    void timerUpdate();

public slots:
    void slot_loadPoissonObj();
    void slot_gravityActivated(bool b);
    void slot_reset();

signals:
    void sig_ResizeToCamera(int,int);
    void sig_DisableGUI(bool);

};


#endif // MYGL_H
