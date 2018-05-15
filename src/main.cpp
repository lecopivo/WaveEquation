#include <Magnum/Buffer.h>
#include <Magnum/DefaultFramebuffer.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Mesh.h>
#include <Magnum/MeshTools/Compile.h>
#include <Magnum/MeshTools/CompressIndices.h>
#include <Magnum/MeshTools/Interleave.h>
#include <Magnum/Platform/Sdl2Application.h>
#include <Magnum/Primitives/Cube.h>
#include <Magnum/Primitives/Icosphere.h>
#include <Magnum/Primitives/Plane.h>
#include <Magnum/Primitives/UVSphere.h>
#include <Magnum/Renderer.h>
#include <Magnum/SceneGraph/Camera.h>
#include <Magnum/SceneGraph/Drawable.h>
#include <Magnum/SceneGraph/MatrixTransformation3D.h>
#include <Magnum/SceneGraph/Scene.h>
#include <Magnum/Shader.h>
#include <Magnum/Shaders/Phong.h>
#include <Magnum/Shaders/VertexColor.h>
#include <Magnum/Trade/MeshData3D.h>

#include <MagnumImGui.h>
#include <imgui.h>

#include <iostream>


#include <Eigen/Dense>

#include "DrawingPrimitives.hpp"

namespace Magnum {

using namespace Magnum::Math::Literals;
using namespace Eigen;

typedef SceneGraph::Object<SceneGraph::MatrixTransformation3D> Object3D;
typedef SceneGraph::Scene<SceneGraph::MatrixTransformation3D>  Scene3D;

struct Settings {
  bool  do_update        = true;
  int   n                = 300;
  float lambda           = 0.5;
  bool  source_on        = false;
  bool  source_move      = false;
  float source_speed     = 1.0;
  float source_frequency = 1.0;
  float time             = 0.f;
  float interface_speed  = 1.0f;
  bool  interface_on     = false;
} sett;

class WaveEquation {
public:
  WaveEquation(int _n) : n(_n), u(n, n), u0(n, n), v(n, n), b(n, n) { reset(); }

  void reset() {

    clearState();
    clearBoundary();
  }

  void clearBoundary() {
    b.resize(n, n);
    b.setOnes();
    for (int i = 0; i < n; i++) {
      b(i, 0)     = 0.0;
      b(0, i)     = 0.0;
      b(i, n - 1) = 0.0;
      b(n - 1, i) = 0.0;
    }
  }

  void harborBoundary() {

    clearBoundary();

    for (int i = 0; i < n; i++) {
      for (int j = n / 2 - 10; j < n / 2 + 10; j++) {
        if (i <= 9 * n / 20 || i > 11 * n / 20) {
          b(j, i) = 0.0;
        }
      }
    }
  }

  void interfaceBoundary(float speed) {

    clearBoundary();

    for (int i = n / 2; i < n; i++) {
      for (int j = 0; j < n; j++) {
        b(i, j) *= speed;
      }
    }
  }

  void clearState() {
    u.resize(n, n);
    u.setZero();
    u0.resize(n, n);
    u0.setZero();
    v.resize(n, n);
    v.setZero();
    v0.resize(n, n);
    v0.setZero();

    sett.time = 0.0f;
  }

  void applyForce(float x, float y, float radius, float strength) {

    x = (n - 1) * 0.5 * (x + 1.0);
    y = (n - 1) * 0.5 * (y + 1.0);

#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        if (b(i, j) != 0.0) {
          float r2  = pow(i - x, 2.0) + pow(j - y, 2.0);
          float rad = 0.5 * (n - 1) * radius;
          v(i, j) += sett.lambda * strength / pow(n / 100.0, 2.0) *
                     exp(-0.5 * r2 / (rad * rad));
        }
      }
    }
  }

  void timesStep(float lambda) {

    u0 = u;
    v0 = v;

    u = u0 + lambda * v;

#pragma omp parallel for collapse(2)
    for (int i = 1; i < n - 1; i++) {
      for (int j = 1; j < n - 1; j++) {
        if (b(i, j) != 0) {
          v(i, j) =
              v0(i, j); // + lambda * b(i, j) *
                        //      (-4 * u(i, j) + u(i + 1, j) + u(i - 1, j) +
                        //       u(i, j + 1) + u(i, j - 1));
          v(i, j) -= lambda * 0.5 *
                     (4.0 * b(i, j) + b(i + 1, j) + b(i - 1, j) + b(i, j + 1) +
                      b(i, j - 1)) *
                     u(i, j);
          v(i, j) += lambda * 0.5 *
                     ((b(i + 1, j) + b(i, j)) * u(i + 1, j) +
                      (b(i, j) + b(i - 1, j)) * u(i - 1, j));
          v(i, j) += lambda * 0.5 *
                     ((b(i, j + 1) + b(i, j)) * u(i, j + 1) +
                      (b(i, j) + b(i, j - 1)) * u(i, j - 1));
        }
      }
    }
  }

  int      n;
  MatrixXf u, u0;
  MatrixXf v, v0;
  MatrixXf b;
};

class PrimitivesExample : public Platform::Application {
public:
  explicit PrimitivesExample(const Arguments &arguments);

private:
  void drawEvent() override;
  void drawGui();
  void update();

  void viewportEvent(const Vector2i &size) override;

  void keyPressEvent(KeyEvent &event) override;
  void keyReleaseEvent(KeyEvent &event) override;
  void mousePressEvent(MouseEvent &event) override;
  void mouseReleaseEvent(MouseEvent &event) override;
  void mouseMoveEvent(MouseMoveEvent &event) override;
  void mouseScrollEvent(MouseScrollEvent &event) override;
  void textInputEvent(TextInputEvent &event) override;

  void mouseRotation(MouseMoveEvent const &event, Vector2 delta);
  void mouseZoom(MouseMoveEvent const &event, Vector2 delta);
  void mousePan(MouseMoveEvent const &event, Vector2 delta);

  Vector3 mousePlanePosition(Vector2i mouseScreenPos);

  Scene3D                     _scene;
  Object3D *                  _cameraObject;
  SceneGraph::Camera3D *      _camera;
  SceneGraph::DrawableGroup3D _drawables;

  Vector2i _previousMousePosition;
  Vector3  _mousePlanePosition[2];
  Vector3  _source;

  MagnumImGui _gui;

  DrawablePlane * plane;
  DrawableSphere *sphere;
  DrawableLine *  line;

  WaveEquation _wave;
  float        time = 0.f;

  Color3 _color;
};

PrimitivesExample::PrimitivesExample(const Arguments &arguments)
    : Platform::Application{arguments,
                            Configuration{}
                                .setTitle("Magnum object picking example")
                                .setWindowFlags(Sdl2Application::Configuration::
                                                    WindowFlag::Resizable)},
      _wave(sett.n) {

  Renderer::enable(Renderer::Feature::DepthTest);
  Renderer::enable(Renderer::Feature::FaceCulling);

  Renderer::setPointSize(12.0);
  Renderer::setLineWidth(6.0);

  //   /* Configure camera */
  _cameraObject = new Object3D{&_scene};
  _cameraObject->translate(Vector3::zAxis(4.0f)).rotateX(Rad{M_PI / 4});
  _camera = new SceneGraph::Camera3D{*_cameraObject};
  viewportEvent(defaultFramebuffer.viewport().size()); // set up camera

  /* TODO: Prepare your objects here and add them to the scene */
  // sphere = (new DrawableSphere(&_scene, &_drawables, 10, 10));

  // sphere->setVertices([](int i, DrawableSphere::VertexData &v) {
  //   v.color = Color4{1.f, 0.f, 0.f, 1.f};
  //   v.position *= 0.f;
  // });

  plane = (new DrawablePlane(&_scene, &_drawables, sett.n - 1, sett.n - 1));

  line = (new DrawableLine(&_scene, &_drawables, 2 * sett.n * sett.n));
}

void PrimitivesExample::update() {

  _wave.timesStep(sett.lambda);
  if (sett.source_on) {
    if (sett.source_move) {
      _wave.applyForce(fmod(sett.source_speed * 0.0015 * sett.time, 1.0) - 0.5f,
                       0.f, 0.005,
                       sin(1.0 / sett.source_frequency * sett.time));
    } else {
      _wave.applyForce(_source[0], _source[1], 0.005,
                       sin(1.0 / sett.source_frequency * sett.time));
    }
  }

  sett.time += sett.lambda;

  // mouse location visualization
  // sphere->translate(
  //     _mousePlanePosition[0] -
  //     sphere->transformation().transformPoint(Vector3{0.f, 0.f, 0.f}));
}

void PrimitivesExample::drawEvent() {
  defaultFramebuffer.clear(FramebufferClear::Color | FramebufferClear::Depth);

  if (sett.do_update) {
    update();
  }

  // Set up wave grid visualization
  plane->setVertices([this](int i, DrawableMesh::VertexData &v) {
    int ix        = i / sett.n;
    int iy        = i % sett.n;
    v.position[2] = _wave.u(ix, iy);
    v.color       = Color4::fromHsv(Rad{10.f * M_PI * v.position[2]}, 0.7, 1.0);
  });

  line->_mesh.setPrimitive(MeshPrimitive::Lines);

  line->setVertices([this](int i, DrawableLine::VertexData &v) {
    int   ix  = (i / 2) / sett.n;
    int   iy  = (i / 2) % sett.n;
    float x   = 2 * (((float)ix) / (sett.n - 1) - 0.5);
    float y   = 2 * (((float)iy) / (sett.n - 1) - 0.5);
    float z   = _wave.u(ix, iy);
    float vel = _wave.v(ix, iy);
    if (i % 2 == 0)
      v.position = Vector3{x, y, z};
    else
      v.position = Vector3{x, y, z + sett.lambda * vel};

    if (ix == sett.n / 2 && iy == sett.n / 2)
      v.color = Color4{1.0, 0., 0., 1.};

    if ((abs(ix - sett.n / 2) + abs(iy - sett.n / 2)) == 1)
      v.color = Color4{0.0, 1., 0., 1.};

    if (plane->_mesh.primitive() == MeshPrimitive::Triangles)
      v.color[3] = 0.0;

  });

  _camera->draw(_drawables);

  drawGui();

  swapBuffers();
}

void PrimitivesExample::drawGui() {
  _gui.newFrame(windowSize(), defaultFramebuffer.viewport().size());

  ImGui::ColorEdit3("Box color", &(_color[0]));

  ImGui::SliderFloat("Time step", &sett.lambda, 0.05, 0.5);

  ImGui::Checkbox("source", &sett.source_on);
  if (sett.source_on) {

    ImGui::SliderFloat("wavelength", &sett.source_frequency, 1.0, 10.0);

    if (ImGui::Checkbox("moving source", &sett.source_move))
      sett.time = 0.0;
    if (sett.source_move) {
      ImGui::SliderFloat("source speed", &sett.source_speed, 1.0, 5.0);
    }
  }

  if (ImGui::Button("Clear boundary")) {
    _wave.clearBoundary();
  }

  if (ImGui::Button("Harbour")) {
    _wave.harborBoundary();
  }

  ImGui::Checkbox("Interface", &sett.interface_on);
  if (sett.interface_on) {
    ImGui::SliderFloat("speed", &sett.interface_speed, 0.5, 2.0);
    _wave.interfaceBoundary(sett.interface_speed);
  }

  if (ImGui::Button("Clear state")) {
    _wave.clearState();
  }

  if (ImGui::Button("Reset")) {
    _wave.reset();
  }

  _gui.drawFrame();

  redraw();
}

void PrimitivesExample::viewportEvent(const Vector2i &size) {
  defaultFramebuffer.setViewport({{}, size});

  _camera->setProjectionMatrix(Matrix4::perspectiveProjection(
      60.0_degf, Vector2{size}.aspectRatio(), 0.001f, 10000.0f));
}

void PrimitivesExample::keyPressEvent(KeyEvent &event) {
  if (_gui.keyPressEvent(event)) {
    redraw();
    return;
  }

  if (event.key() == KeyEvent::Key::Esc) {
    exit();
  }

  if (event.key() == KeyEvent::Key::Space) {
    _source = _mousePlanePosition[0];
  }

  if (event.key() == KeyEvent::Key::S) {
    sett.do_update = !sett.do_update;
  }

  if (event.key() == KeyEvent::Key::C) {
    _wave.clearState();
  }

  if (event.key() == KeyEvent::Key::D) {
    update();
  }

  if (event.key() == KeyEvent::Key::W) {

    if (plane->_mesh.primitive() == MeshPrimitive::Points) {
      plane->_mesh.setPrimitive(MeshPrimitive::Triangles);
    } else {
      plane->_mesh.setPrimitive(MeshPrimitive::Points);
    }
  }

  redraw();
}

void PrimitivesExample::keyReleaseEvent(KeyEvent &event) {
  if (_gui.keyReleaseEvent(event)) {
    redraw();
    return;
  }
}

void PrimitivesExample::mousePressEvent(MouseEvent &event) {
  if (_gui.mousePressEvent(event)) {
    redraw();
    return;
  }

  if (event.button() != MouseEvent::Button::Left)
    return;

  _previousMousePosition = event.position();
  event.setAccepted();
}

void PrimitivesExample::mouseReleaseEvent(MouseEvent &event) {
  if (_gui.mouseReleaseEvent(event)) {
    redraw();
    return;
  }

  event.setAccepted();
  redraw();
}

void PrimitivesExample::mouseMoveEvent(MouseMoveEvent &event) {
  if (_gui.mouseMoveEvent(event)) {
    redraw();
    return;
  }

  float lambda = 0.5;
  _mousePlanePosition[1] =
      lambda * _mousePlanePosition[0] + (1 - lambda) * _mousePlanePosition[1];
  _mousePlanePosition[0] = mousePlanePosition(event.position());

  const Vector2 delta = Vector2{event.position() - _previousMousePosition} /
                        Vector2{defaultFramebuffer.viewport().size()};

  if (event.buttons() & MouseMoveEvent::Button::Left)
    mouseRotation(event, delta);

  if (event.buttons() & MouseMoveEvent::Button::Right)
    mouseZoom(event, delta);

  if (event.buttons() & MouseMoveEvent::Button::Middle)
    _wave.applyForce(_mousePlanePosition[0][0], _mousePlanePosition[0][1], 0.02,
                     0.1);
  // mousePan(event, delta);

  _previousMousePosition = event.position();
  event.setAccepted();
  redraw();
}

void PrimitivesExample::mouseScrollEvent(MouseScrollEvent &event) {
  if (_gui.mouseScrollEvent(event)) {
    redraw();
    return;
  }
}

void PrimitivesExample::textInputEvent(TextInputEvent &event) {
  if (_gui.textInputEvent(event)) {
    redraw();
    return;
  }
}

void PrimitivesExample::mouseRotation(MouseMoveEvent const &event,
                                      Vector2               delta) {

  auto camPos =
      _cameraObject->transformation().transformPoint(Vector3{0.0, 0.0, 0.0});

  auto axis = cross(Vector3{0.f, 0.f, 1.f}, camPos.normalized()).normalized();

  _cameraObject->rotate(Rad{-3.0f * delta.y()}, axis);
  _cameraObject->rotateZ(Rad{-3.0f * delta.x()});
}

void PrimitivesExample::mouseZoom(MouseMoveEvent const &event, Vector2 delta) {
  auto dir =
      _cameraObject->transformation().transformVector(Vector3{0.0, 0.0, 1.0});

  _cameraObject->translate(10.0f * delta.y() * dir);
}

void PrimitivesExample::mousePan(MouseMoveEvent const &event, Vector2 delta) {}

Vector3 PrimitivesExample::mousePlanePosition(Vector2i mouseScreenPos) {
  Vector2 mpos = 2.0f * (Vector2{mouseScreenPos} /
                             Vector2{defaultFramebuffer.viewport().size()} -
                         Vector2{.5f, 0.5f});
  mpos[1] *= -1.f;

  Vector3 mpos3 = {mpos[0], mpos[1], -1.f};
  auto    trans =
      _cameraObject->transformation() * _camera->projectionMatrix().inverted();
  mpos3 = trans.transformPoint(mpos3);
  Vector3 camOrg =
      _cameraObject->transformation().transformPoint(Vector3{0.f, 0.f, 0.f});
  Vector3 dir    = mpos3 - camOrg;
  float   lambda = -camOrg.z() / dir.z();
  mpos3          = camOrg + lambda * dir;

  return mpos3;
}

} // namespace Magnum

MAGNUM_APPLICATION_MAIN(Magnum::PrimitivesExample)
