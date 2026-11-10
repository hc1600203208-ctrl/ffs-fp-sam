#include <Eigen/Dense>
#include <png.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <future>
#include <cstring>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/select.h>
#include <sys/socket.h>
#include "sct_protocol.hpp"
#include "udp_wire.hpp"

namespace {
using Clock = std::chrono::steady_clock;
struct Face { std::array<int, 3> v, uv; Eigen::Vector3f color; };
struct Mesh { std::vector<Eigen::Vector3f> vertices; std::vector<Eigen::Vector2f> uv; std::vector<Face> faces; std::vector<uint8_t> texture; int tex_w=0,tex_h=0; };
struct Camera { float fx, fy, cx, cy; Eigen::Matrix3f rotation; Eigen::Vector3f translation; };
struct Prepared { std::vector<Eigen::Vector3f> points; std::vector<Eigen::Vector2f> projected; };
constexpr float pi = 3.14159265358979323846f;
uint32_t index(const std::string& token, size_t count) {
  int n = std::stoi(token);
  int i = n > 0 ? n-1 : static_cast<int>(count)+n;
  if (i < 0 || i >= static_cast<int>(count)) throw std::runtime_error("OBJ face index out of range");
  return static_cast<uint32_t>(i);
}
Mesh load_mesh(const std::string& path) {
  std::ifstream in(path); if (!in) throw std::runtime_error("cannot open OBJ: " + path);
  Mesh mesh; std::string line; Eigen::Vector3f color(.72f,.72f,.72f);
  std::string directory=path.substr(0,path.find_last_of("/\\")+1);
  while (std::getline(in,line)) {
    std::istringstream s(line); std::string tag; s>>tag;
    if (tag=="v") { Eigen::Vector3f v; if (s>>v.x()>>v.y()>>v.z()) mesh.vertices.push_back(v); }
    else if (tag=="vt") { Eigen::Vector2f v; if(s>>v.x()>>v.y()) mesh.uv.push_back(v); }
    else if (tag=="mtllib") {
      std::string name; s>>name;std::ifstream mtl(directory+name);std::string entry;
      while(std::getline(mtl,entry)) {
        std::istringstream ms(entry);std::string key;ms>>key;
        if(key=="map_Kd") {std::string file;ms>>file;png_image png{};png.version=PNG_IMAGE_VERSION;
          if(png_image_begin_read_from_file(&png,(directory+file).c_str())) {
            png.format=PNG_FORMAT_RGB;mesh.texture.resize(PNG_IMAGE_SIZE(png));
            if(png_image_finish_read(&png,nullptr,mesh.texture.data(),0,nullptr)) {
              mesh.tex_w=png.width;mesh.tex_h=png.height;
            } else {
              mesh.texture.clear();
            }
            png_image_free(&png);
          }
        }
      }
    }
    else if (tag=="f") {
      std::vector<int> ids,tex; std::string t;
      while (s>>t) {auto slash=t.find('/');ids.push_back(index(t.substr(0,slash),mesh.vertices.size()));
        auto end=t.find('/',slash+1);tex.push_back(slash!=std::string::npos&&end>slash+1?
          index(t.substr(slash+1,end-slash-1),mesh.uv.size()):0);}
      for (size_t i=1;i+1<ids.size();++i) mesh.faces.push_back({{ids[0],ids[i],ids[i+1]},
        {tex[0],tex[i],tex[i+1]},color});
    }
  }
  if (mesh.vertices.empty() || mesh.faces.empty()) throw std::runtime_error("OBJ has no triangles: " + path);
  return mesh;
}
std::vector<double> numbers(const std::string& path) {
  std::ifstream in(path); if (!in) throw std::runtime_error("cannot open calibration: " + path);
  std::vector<double> out; std::string line;
  while (std::getline(in,line)) { if (line.empty() || line[0]=='#') continue;
    std::istringstream s(line); double n; while (s>>n) out.push_back(n); }
  if (out.size()<41) throw std::runtime_error("incomplete stereo calibration");
  return out;
}
Eigen::Matrix3f euler(float roll,float pitch,float yaw) {
  return (Eigen::AngleAxisf(yaw,Eigen::Vector3f::UnitZ()) *
          Eigen::AngleAxisf(pitch,Eigen::Vector3f::UnitY()) *
          Eigen::AngleAxisf(roll,Eigen::Vector3f::UnitX())).toRotationMatrix();
}
void draw(const Mesh& mesh, const Eigen::Matrix3f& rotation, const Eigen::Vector3f& position,
          const Camera& camera,int width,int height,std::vector<uint16_t>& pixels,std::vector<float>& depth) {
  const Eigen::Vector3f light=Eigen::Vector3f(.3f,-.5f,-1.f).normalized();
  Prepared prepared;prepared.points.reserve(mesh.vertices.size());prepared.projected.reserve(mesh.vertices.size());
  const Eigen::Matrix3f transform=camera.rotation*rotation;
  const Eigen::Vector3f offset=camera.rotation*position+camera.translation;
  for(const auto& vertex:mesh.vertices){
    Eigen::Vector3f p=transform*vertex+offset;prepared.points.push_back(p);
    prepared.projected.emplace_back(camera.fx*p.x()/p.z()+camera.cx,camera.fy*p.y()/p.z()+camera.cy);
  }
  for (const Face& face:mesh.faces) {
    Eigen::Vector3f p[3]; Eigen::Vector2f uv[3]; bool visible=true;
    for (int i=0;i<3;++i) {
      p[i]=prepared.points[face.v[i]];
      if (p[i].z()<=.05f) { visible=false; break; }
      uv[i]=prepared.projected[face.v[i]];
    }
    if (!visible) continue;
    float area=(uv[1].x()-uv[0].x())*(uv[2].y()-uv[0].y())-(uv[1].y()-uv[0].y())*(uv[2].x()-uv[0].x());
    if (std::abs(area)<1e-5f) continue;
    int x0=std::max(0,static_cast<int>(std::floor(std::min({uv[0].x(),uv[1].x(),uv[2].x()}))));
    int x1=std::min(width-1,static_cast<int>(std::ceil(std::max({uv[0].x(),uv[1].x(),uv[2].x()}))));
    int y0=std::max(0,static_cast<int>(std::floor(std::min({uv[0].y(),uv[1].y(),uv[2].y()}))));
    int y1=std::min(height-1,static_cast<int>(std::ceil(std::max({uv[0].y(),uv[1].y(),uv[2].y()}))));
    if (x0>x1 || y0>y1) continue;
    float inv_area=1.f/area;
    float shade=.25f+.75f*std::abs(((p[1]-p[0]).cross(p[2]-p[0])).normalized().dot(light));
    auto channel=[&](int c,int bits) { return static_cast<uint16_t>(std::clamp(face.color[c]*shade,0.f,1.f)*((1<<bits)-1)); };
    uint16_t rgb=(channel(0,5)<<11)|(channel(1,6)<<5)|channel(2,5);
    for (int y=y0;y<=y1;++y) for (int x=x0;x<=x1;++x) {
      float px=x+.5f,py=y+.5f;
      float a=((uv[1].x()-px)*(uv[2].y()-py)-(uv[1].y()-py)*(uv[2].x()-px))*inv_area;
      float b=((uv[2].x()-px)*(uv[0].y()-py)-(uv[2].y()-py)*(uv[0].x()-px))*inv_area;
      float c=1-a-b; if (a<0 || b<0 || c<0) continue;
      float z=1.f/(a/p[0].z()+b/p[1].z()+c/p[2].z());
      size_t at=static_cast<size_t>(y)*width+x;
      if (z<depth[at]) {
        depth[at]=z;
        if(mesh.texture.empty() || mesh.uv.empty()) pixels[at]=rgb;
        else {
          Eigen::Vector2f uvp=(a*mesh.uv[face.uv[0]]/p[0].z()+b*mesh.uv[face.uv[1]]/p[1].z()+
            c*mesh.uv[face.uv[2]]/p[2].z())*z;
          int tx=std::clamp(int(uvp.x()*mesh.tex_w),0,mesh.tex_w-1);
          int ty=std::clamp(int((1-uvp.y())*mesh.tex_h),0,mesh.tex_h-1);
          size_t ti=(size_t(ty)*mesh.tex_w+tx)*3;
          auto sample=[&](int component,int bits){return uint16_t(std::clamp(mesh.texture[ti+component]*shade,0.f,255.f)*(1<<bits)/256);};
          pixels[at]=(sample(0,5)<<11)|(sample(1,6)<<5)|sample(2,5);
        }
      }
    }
  }
}
}

namespace {
void put_double(std::vector<std::uint8_t>& bytes, double value) {
  std::uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  udp_wire::put64(bytes, bits);
}
std::string asset_path(const std::string& root, const std::string& name) {
  const auto path = std::filesystem::path(name);
  if (path.empty() || path.is_absolute()) throw std::invalid_argument("asset path must be relative");
  for (const auto& component : path)
    if (component == "..") throw std::invalid_argument("asset path cannot leave the bundle");
  return (std::filesystem::path(root) / path).string();
}
std::vector<std::uint8_t> encode_png(const std::vector<std::uint16_t>& pixels, int width, int height) {
  std::vector<std::uint8_t> rgb(std::size_t(width) * height * 2 * 3);
  for (int eye = 0; eye < 2; ++eye) {
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
      auto color = pixels[(std::size_t(eye) * height + y) * width + x];
      auto at = (std::size_t(y) * width * 2 + eye * width + x) * 3;
      rgb[at] = ((color >> 11) & 31) * 255 / 31;
      rgb[at + 1] = ((color >> 5) & 63) * 255 / 63;
      rgb[at + 2] = (color & 31) * 255 / 31;
    }
  }
  png_image image{};
  image.version = PNG_IMAGE_VERSION;
  image.width = width * 2; image.height = height; image.format = PNG_FORMAT_RGB;
  png_alloc_size_t length = 0;
  if (!png_image_write_to_memory(&image, nullptr, &length, 0, rgb.data(), 0, nullptr))
    throw std::runtime_error("PNG size calculation failed");
  std::vector<std::uint8_t> output(length);
  if (!png_image_write_to_memory(&image, output.data(), &length, 0, rgb.data(), 0, nullptr))
    throw std::runtime_error("PNG encoding failed");
  output.resize(length);
  return output;
}
}

class SatelliteRendererUdp {
 public:
  SatelliteRendererUdp(std::uint16_t port, std::string assets) : assets_(std::move(assets)) {
    fd_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd_ < 0) throw std::runtime_error("cannot open UDP socket");
    sockaddr_in bind_to{};
    bind_to.sin_family = AF_INET; bind_to.sin_addr.s_addr = htonl(INADDR_ANY);
    bind_to.sin_port = htons(port);
    if (bind(fd_, reinterpret_cast<sockaddr*>(&bind_to), sizeof(bind_to)) != 0)
      throw std::runtime_error("cannot bind UDP port");
    std::fprintf(stderr, "Satellite UDP renderer listening on %u; assets=%s\n", port, assets_.c_str());
  }
  ~SatelliteRendererUdp() { if (fd_ >= 0) close(fd_); }

  void run() {
    while (true) {
      fd_set ready; FD_ZERO(&ready); FD_SET(fd_, &ready);
      timeval timeout{0, 100000};
      if (running_) {
        auto wait = std::chrono::duration_cast<std::chrono::microseconds>(next_frame_ - Clock::now());
        auto us = std::clamp<std::int64_t>(wait.count(), 0, 100000);
        timeout.tv_sec = us / 1000000; timeout.tv_usec = us % 1000000;
      }
      int count = select(fd_ + 1, &ready, nullptr, nullptr, &timeout);
      if (count > 0 && FD_ISSET(fd_, &ready)) receive_control();
      if (running_ && Clock::now() >= next_frame_) {
        try { render(); }
        catch (const std::exception& e) { std::fprintf(stderr, "render error: %s\n", e.what()); running_ = false; }
        next_frame_ = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(1.0 / fps_));
      }
      reassembler_.expire();
    }
  }

 private:
  void receive_control() {
    std::uint8_t buffer[udp_wire::header_size + udp_wire::chunk_size];
    sockaddr_in from{}; socklen_t from_len = sizeof(from);
    auto size = recvfrom(fd_, buffer, sizeof(buffer), 0, reinterpret_cast<sockaddr*>(&from), &from_len);
    udp_wire::Message message{};
    if (size <= 0 || !reassembler_.push(buffer, size, from.sin_addr.s_addr, from.sin_port, &message) ||
        message.kind != udp_wire::control) return;
    const auto key = std::make_tuple(from.sin_addr.s_addr, from.sin_port, message.id);
    if (auto previous = acknowledgements_.find(key); previous != acknowledgements_.end()) {
      udp_wire::send(fd_, from, {udp_wire::ack, message.id, 0, previous->second});
      return;
    }
    std::uint32_t id = message.payload.size() >= 12 ? udp_wire::get32(message.payload.data() + 8) : message.id;
    std::uint8_t status = 0; std::string detail = "ok";
    try {
      auto request = sct::parse(message.payload);
      if (request.id != message.id) throw std::invalid_argument("request id mismatch");
      if (request.type == 1) {
        if (running_) throw std::invalid_argument("stop before reconfiguration");
        configure(request.config);
      } else if (request.type == 2) {
        if (!configured_) throw std::invalid_argument("configure before start");
        started_ = Clock::now(); next_frame_ = started_; next_send_ = started_; frame_ = 0; running_ = true;
      } else if (request.type == 3) running_ = false;
    } catch (const std::exception& e) { status = 1; detail = e.what(); }
    auto acknowledgement = sct::ack(id, status, detail);
    acknowledgements_[key] = acknowledgement;
    if (acknowledgements_.size() > 64) acknowledgements_.erase(acknowledgements_.begin());
    udp_wire::send(fd_, from, {udp_wire::ack, id, 0, std::move(acknowledgement)});
    if (!status) peer_ = from;
    std::fprintf(stderr, "control %u status=%u %s\n", id, status, detail.c_str());
  }

  void configure(const sct::Config& config) {
    if (config.width == 0 || config.width > 4096 || config.height == 0 || config.height > 4096 ||
        config.fps <= 0 || config.fps > 1000 || !std::isfinite(config.fps) ||
        config.bytes_per_second == 0 || config.objects.empty()) throw std::invalid_argument("invalid renderer configuration");
    auto n = numbers(asset_path(assets_, config.calibration_path));
    std::array<Camera, 2> cameras;
    cameras[0] = {float(n[0]), float(n[4]), float(n[2]), float(n[5]), Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero()};
    cameras[1] = {float(n[15]), float(n[19]), float(n[17]), float(n[20]), Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero()};
    for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) cameras[1].rotation(r, c) = n[29 + r * 3 + c];
    for (int i = 0; i < 3; ++i) cameras[1].translation[i] = n[38 + i];
    std::vector<Mesh> meshes;
    for (const auto& object : config.objects) meshes.push_back(load_mesh(asset_path(assets_, object.mesh_path)));
    cameras_ = std::move(cameras); meshes_ = std::move(meshes); objects_ = config.objects;
    width_ = config.width; height_ = config.height; fps_ = config.fps; rate_ = config.bytes_per_second;
    configured_ = true;
    std::fprintf(stderr, "configured %zu objects %dx%d %.2f fps\n", objects_.size(), width_, height_, fps_);
  }

  void render() {
    auto t = std::chrono::duration<double>(Clock::now() - started_).count();
    auto stamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    std::vector<Eigen::Vector3f> positions;
    std::vector<Eigen::Matrix3f> rotations;
    std::vector<std::uint8_t> truth_payload;
    udp_wire::put16(truth_payload, objects_.size());
    for (const auto& object : objects_) {
      Eigen::Vector3f position, angles;
      for (int axis = 0; axis < 3; ++axis) {
        position[axis] = object.pose[axis] + object.amplitudes[axis] * std::sin(2 * pi * object.frequencies[axis] * t);
        angles[axis] = float(object.pose[axis + 3] + object.amplitudes[axis + 3] *
                             std::sin(2 * pi * object.frequencies[axis + 3] * t)) * pi / 180;
      }
      auto rotation = euler(angles[0], angles[1], angles[2]);
      Eigen::Quaternionf q(rotation); q.normalize();
      for (int axis = 0; axis < 3; ++axis) put_double(truth_payload, position[axis]);
      for (double component : {double(q.x()), double(q.y()), double(q.z()), double(q.w())}) put_double(truth_payload, component);
      positions.push_back(position); rotations.push_back(rotation);
    }
    std::vector<std::uint16_t> pixels(2 * std::size_t(width_) * height_, 0);
    auto eye = [&](int index) {
      std::vector<std::uint16_t> image(std::size_t(width_) * height_, 0);
      std::vector<float> depth(image.size(), std::numeric_limits<float>::infinity());
      for (std::size_t j = 0; j < meshes_.size(); ++j)
        draw(meshes_[j], rotations[j], positions[j], cameras_[index], width_, height_, image, depth);
      std::copy(image.begin(), image.end(), pixels.begin() + index * image.size());
    };
    auto right = std::async(std::launch::async, eye, 1); eye(0); right.get();
    auto png = encode_png(pixels, width_, height_);
    udp_wire::send(fd_, peer_, {udp_wire::truth, frame_, std::uint64_t(stamp), std::move(truth_payload)});
    for (const auto& chunk : udp_wire::encode({udp_wire::stereo_png, frame_, std::uint64_t(stamp), std::move(png)})) {
      auto due = next_send_;
      if (due > Clock::now()) std::this_thread::sleep_until(due);
      auto sent = sendto(fd_, chunk.data(), chunk.size(), 0, reinterpret_cast<const sockaddr*>(&peer_), sizeof(peer_));
      if (sent != static_cast<ssize_t>(chunk.size())) throw std::runtime_error("UDP frame send failed");
      next_send_ = std::max(due, Clock::now()) +
          std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(chunk.size() / double(rate_)));
    }
    if (frame_++ % 30 == 0) std::fprintf(stderr, "frame %u transmitted\n", frame_);
  }

  int fd_{-1}, width_{0}, height_{0};
  std::uint32_t frame_{0}, rate_{0};
  double fps_{0}; bool configured_{false}, running_{false};
  std::string assets_;
  sockaddr_in peer_{};
  udp_wire::Reassembler reassembler_;
  std::map<std::tuple<std::uint32_t, std::uint16_t, std::uint32_t>,
           std::vector<std::uint8_t>> acknowledgements_;
  std::array<Camera, 2> cameras_{};
  std::vector<Mesh> meshes_;
  std::vector<sct::Object> objects_;
  Clock::time_point started_{}, next_frame_{}, next_send_{};
};

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s <udp_port> <asset_directory>\n", argv[0]);
    return 2;
  }
  try { SatelliteRendererUdp(std::stoi(argv[1]), argv[2]).run(); }
  catch (const std::exception& e) { std::fprintf(stderr, "satellite renderer: %s\n", e.what()); return 1; }
}
