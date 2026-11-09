#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/u_int8_multi_array.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <Eigen/Dense>
#include <png.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <future>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "sct_protocol.hpp"

namespace {
using Clock = std::chrono::steady_clock;
struct Face { std::array<int, 3> v, uv; Eigen::Vector3f color; };
struct Mesh { std::vector<Eigen::Vector3f> vertices; std::vector<Eigen::Vector2f> uv; std::vector<Face> faces; std::vector<uint8_t> texture; int tex_w=0,tex_h=0; };
struct Camera { float fx, fy, cx, cy; Eigen::Matrix3f rotation; Eigen::Vector3f translation; };
struct Prepared { std::vector<Eigen::Vector3f> points; std::vector<Eigen::Vector2f> projected; };
constexpr float pi = 3.14159265358979323846f;
void put32(std::vector<uint8_t>& b, uint32_t n) { for (int i=0;i<4;++i) b.push_back((n>>(8*i))&255); }
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
            if(png_image_finish_read(&png,nullptr,mesh.texture.data(),0,nullptr)) {mesh.tex_w=png.width;mesh.tex_h=png.height;}
            else mesh.texture.clear();png_image_free(&png);
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

class SatelliteRenderer: public rclcpp::Node {
 public:
  SatelliteRenderer():Node("satellite_render_node") {
    auto paths=declare_parameter<std::vector<std::string>>("mesh_paths",std::vector<std::string>{});
    auto poses=declare_parameter<std::vector<double>>("initial_poses",std::vector<double>{});
    auto amp=declare_parameter<std::vector<double>>("amplitudes",std::vector<double>(6,0));
    auto freq=declare_parameter<std::vector<double>>("frequencies",std::vector<double>(6,0));
    auto width=declare_parameter<int>("width",640),height=declare_parameter<int>("height",480);
    auto fps=declare_parameter<double>("fps",2.0);
    auto rate=declare_parameter<int>("bytes_per_second",2000000);
    auto calibration=declare_parameter<std::string>("calibration_path","");
    publisher_=create_publisher<std_msgs::msg::UInt8MultiArray>("/satellite/stereo_bytes",128);
    truth_publisher_=create_publisher<geometry_msgs::msg::PoseArray>("/satellite/ground_truth",10);
    ack_publisher_=create_publisher<std_msgs::msg::UInt8MultiArray>("/satellite/control_ack_bytes",10);
    control_sub_=create_subscription<std_msgs::msg::UInt8MultiArray>("/satellite/control_bytes",10,
      [this](std_msgs::msg::UInt8MultiArray::ConstSharedPtr msg){handle_control(msg->data);});
    if(!paths.empty()) {
      if(poses.size()!=paths.size()*6 || amp.size()!=6 || freq.size()!=6)
        throw std::invalid_argument("legacy parameter array lengths do not match meshes");
      sct::Config config{};
      config.width=width;config.height=height;config.fps=fps;config.bytes_per_second=rate;
      config.calibration_path=calibration;
      for(size_t i=0;i<paths.size();++i){sct::Object object;
        object.name="object_"+std::to_string(i);object.mesh_path=paths[i];
        std::copy_n(poses.begin()+i*6,6,object.pose.begin());
        std::copy(amp.begin(),amp.end(),object.amplitudes.begin());
        std::copy(freq.begin(),freq.end(),object.frequencies.begin());
        config.objects.push_back(std::move(object));}
      apply_config(config);start();
    } else RCLCPP_INFO(get_logger(),"Waiting for SCT1 configuration on /satellite/control_bytes");
  }
 private:
  void apply_config(const sct::Config& config) {
    if(config.width==0||config.height==0||config.width>4096||config.height>4096||
       !std::isfinite(config.fps)||config.fps<=0||config.fps>1000||config.bytes_per_second==0||
       config.bytes_per_second>static_cast<uint32_t>(std::numeric_limits<int>::max())||
       config.objects.empty()) throw std::invalid_argument("invalid rendering parameters");
    auto finite=[](double n){return std::isfinite(n);};
    for(const auto& object:config.objects)
      if(object.mesh_path.empty()||!std::all_of(object.pose.begin(),object.pose.end(),finite)||
         !std::all_of(object.amplitudes.begin(),object.amplitudes.end(),finite)||
         !std::all_of(object.frequencies.begin(),object.frequencies.end(),finite)||
         std::any_of(object.amplitudes.begin(),object.amplitudes.end(),[](double n){return n<0;})||
         std::any_of(object.frequencies.begin(),object.frequencies.end(),[](double n){return n<0;}))
        throw std::invalid_argument("invalid object parameters");
    auto n=numbers(config.calibration_path);
    std::array<Camera,2> cameras;
    cameras[0]={float(n[0]),float(n[4]),float(n[2]),float(n[5]),Eigen::Matrix3f::Identity(),Eigen::Vector3f::Zero()};
    cameras[1]={float(n[15]),float(n[19]),float(n[17]),float(n[20]),Eigen::Matrix3f::Identity(),Eigen::Vector3f::Zero()};
    for(int r=0;r<3;++r) for(int c=0;c<3;++c) cameras[1].rotation(r,c)=n[29+r*3+c];
    for(int i=0;i<3;++i) cameras[1].translation[i]=n[38+i];
    std::vector<Mesh> meshes;std::vector<sct::Object> objects;
    for(const auto& object:config.objects){meshes.push_back(load_mesh(object.mesh_path));
      objects.push_back(object);}
    cameras_=std::move(cameras);meshes_=std::move(meshes);objects_=std::move(objects);
    width_=config.width;height_=config.height;fps_=config.fps;rate_=config.bytes_per_second;
    frame_=0;configured_=true;
    RCLCPP_INFO(get_logger(),"Configured %zu objects, %dx%d at %.2f FPS, %d bytes/s",
      meshes_.size(),width_,height_,fps_,rate_);
  }
  void start() {
    started_=Clock::now();next_send_=started_;running_=true;
    if(timer_) timer_->cancel();
    timer_=create_wall_timer(std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>(1/fps_)),[this]{render();});
  }
  void handle_control(const std::vector<uint8_t>& bytes) {
    if(bytes.size()<12 || bytes[0]!='S'||bytes[1]!='C'||bytes[2]!='T'||bytes[3]!='1') return;
    const uint32_t id=uint32_t(bytes[8])|(uint32_t(bytes[9])<<8)|
      (uint32_t(bytes[10])<<16)|(uint32_t(bytes[11])<<24);
    uint8_t status=0;std::string result="ok";
    try {
      auto request=sct::parse(bytes);
      if(request.type==1) {
        if(running_) {status=3;result="stop rendering before configuration";}
        else apply_config(request.config);
      } else if(request.type==2) {
        if(!configured_) {status=3;result="configure before start";}
        else if(!running_) start();
      } else if(request.type==3) {
        if(timer_) timer_->cancel();running_=false;
      }
    } catch(const std::invalid_argument& error) {status=1;result=error.what();}
      catch(const std::exception& error) {status=2;result=error.what();}
    std_msgs::msg::UInt8MultiArray reply;reply.data=sct::ack(id,status,result);
    ack_publisher_->publish(reply);
    RCLCPP_INFO(get_logger(),"SCT1 request %u: status %u, %s",id,status,result.c_str());
  }
  void render() {
    auto frame_start=Clock::now();
    double t=std::chrono::duration<double>(Clock::now()-started_).count();
    std::vector<uint16_t> pixels(2*size_t(width_)*height_,0);
    auto stamp=now();
    geometry_msgs::msg::PoseArray truth;
    truth.header.stamp=stamp;
    truth.header.frame_id="stereo_left_optical_frame";
    std::vector<Eigen::Vector3f> positions;
    std::vector<Eigen::Matrix3f> rotations;
    positions.reserve(objects_.size());rotations.reserve(objects_.size());truth.poses.reserve(objects_.size());
    for(const auto& object:objects_) {
      Eigen::Vector3f pos,angles;
      for(int axis=0;axis<3;++axis) {
        pos[axis]=object.pose[axis]+object.amplitudes[axis]*std::sin(2*pi*object.frequencies[axis]*t);
        angles[axis]=float(object.pose[axis+3]+object.amplitudes[axis+3]*std::sin(2*pi*object.frequencies[axis+3]*t))*pi/180;
      }
      auto rotation=euler(angles[0],angles[1],angles[2]);
      Eigen::Quaternionf q(rotation);q.normalize();
      geometry_msgs::msg::Pose pose;
      pose.position.x=pos.x();pose.position.y=pos.y();pose.position.z=pos.z();
      pose.orientation.x=q.x();pose.orientation.y=q.y();pose.orientation.z=q.z();pose.orientation.w=q.w();
      positions.push_back(pos);rotations.push_back(rotation);truth.poses.push_back(pose);
    }
    auto render_eye=[&](int eye) {
      std::vector<uint16_t> image(size_t(width_)*height_,0);
      std::vector<float> depth(image.size(),std::numeric_limits<float>::infinity());
      for(size_t j=0;j<meshes_.size();++j) {
        draw(meshes_[j],rotations[j],positions[j],cameras_[eye],width_,height_,image,depth);
      }
      std::copy(image.begin(),image.end(),pixels.begin()+eye*image.size());
    };
    auto right_eye=std::async(std::launch::async,render_eye,1);
    render_eye(0);
    right_eye.get();
    auto raster_done=Clock::now();
    // Larger packets avoid hundreds of DDS publishes for every stereo pair.
    constexpr size_t chunk=32768; uint32_t count=(pixels.size()*2+chunk-1)/chunk;
    truth_publisher_->publish(truth);
    auto stamp_ns=stamp.nanoseconds();
    for(uint32_t part=0;part<count;++part) {
      std_msgs::msg::UInt8MultiArray msg; auto& b=msg.data;
      b.reserve(32+chunk); b.insert(b.end(),{'S','T','R','1'});
      put32(b,frame_);put32(b,part);put32(b,count);put32(b,width_);put32(b,height_);
      put32(b,static_cast<uint32_t>(stamp_ns/1000000000));put32(b,static_cast<uint32_t>(stamp_ns%1000000000));
      size_t begin=part*chunk,end=std::min(begin+chunk,pixels.size()*2);
      for(size_t i=begin;i<end;++i) b.push_back((pixels[i/2]>>(8*(i%2)))&255);
      auto due=next_send_;
      if(due>Clock::now()) std::this_thread::sleep_until(due);
      publisher_->publish(msg);
      next_send_=std::max(due,Clock::now())+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(b.size()/double(rate_)));
    }
    ++frame_;
    if(frame_==1 || frame_%30==0) RCLCPP_INFO(get_logger(),"frame %u: raster %.1f ms, transmit %.1f ms, link %.1f MB/s",
      frame_,std::chrono::duration<double,std::milli>(raster_done-frame_start).count(),
      std::chrono::duration<double,std::milli>(Clock::now()-raster_done).count(),rate_/1e6);
  }
  int width_=0,height_=0,rate_=0;double fps_=0;uint32_t frame_=0;
  bool configured_=false,running_=false;
  std::array<Camera,2> cameras_;std::vector<Mesh> meshes_;std::vector<sct::Object> objects_;
  Clock::time_point started_,next_send_;rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<std_msgs::msg::UInt8MultiArray>::SharedPtr publisher_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr truth_publisher_;
  rclcpp::Publisher<std_msgs::msg::UInt8MultiArray>::SharedPtr ack_publisher_;
  rclcpp::Subscription<std_msgs::msg::UInt8MultiArray>::SharedPtr control_sub_;
};
int main(int argc,char** argv) {rclcpp::init(argc,argv);try{rclcpp::spin(std::make_shared<SatelliteRenderer>());}catch(const std::exception& e){fprintf(stderr,"satellite renderer: %s\n",e.what());rclcpp::shutdown();return 1;}rclcpp::shutdown();return 0;}
