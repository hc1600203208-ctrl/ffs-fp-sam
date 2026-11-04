#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/u_int8_multi_array.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <png.h>

namespace {
uint32_t read32(const std::vector<uint8_t>& b,size_t at) {
  return uint32_t(b[at]) | (uint32_t(b[at+1])<<8) | (uint32_t(b[at+2])<<16) | (uint32_t(b[at+3])<<24);
}
std::vector<double> numbers(const std::string& path) {
  std::ifstream in(path);if(!in) throw std::runtime_error("cannot open calibration: "+path);
  std::vector<double> n;std::string line;
  while(std::getline(in,line)){if(line.empty()||line[0]=='#')continue;std::istringstream s(line);double x;while(s>>x)n.push_back(x);}
  if(n.size()<41)throw std::runtime_error("incomplete stereo calibration");return n;
}
std::vector<uint8_t> load_background(const std::string& path,int& width,int& height) {
  png_image image{};image.version=PNG_IMAGE_VERSION;
  if(!png_image_begin_read_from_file(&image,path.c_str())) throw std::runtime_error("cannot read background PNG: "+path);
  image.format=PNG_FORMAT_RGB;width=image.width;height=image.height;
  std::vector<uint8_t> rgb(PNG_IMAGE_SIZE(image));
  if(!png_image_finish_read(&image,nullptr,rgb.data(),0,nullptr)) {
    png_image_free(&image);throw std::runtime_error("cannot decode background PNG: "+path);
  }
  png_image_free(&image);return rgb;
}
}
class Receiver:public rclcpp::Node {
 public:
  Receiver():Node("stereo_bytes_to_images_node") {
    auto n=numbers(declare_parameter<std::string>("calibration_path",""));
    const auto path=declare_parameter<std::string>("background_path","");
    background_disparity_=declare_parameter<int>("background_disparity_px",8);
    if(background_disparity_<0||background_disparity_>128) throw std::runtime_error("background_disparity_px must be 0..128");
    if(!path.empty()) {
      background_=load_background(path,background_width_,background_height_);
      RCLCPP_INFO(get_logger(),"Background %s; fixed disparity %d px (nominal depth %.3f m)",
        path.c_str(),background_disparity_,n[0]*n[9]/std::max(1,background_disparity_));
    }
    for(int eye=0;eye<2;++eye){
      const std::string side=eye?"right":"left";
      images_[eye]=create_publisher<sensor_msgs::msg::Image>("/"+side+"/image_raw",10);
      infos_[eye]=create_publisher<sensor_msgs::msg::CameraInfo>("/"+side+"/camera_info",10);
      size_t k=eye?15:0,d=eye?24:10;
      auto& info=info_[eye];info.distortion_model="plumb_bob";info.header.frame_id="stereo_"+side+"_optical_frame";
      for(int i=0;i<9;++i){info.k[i]=n[k+i];info.r[i]=i%4==0?1:0;}
      for(int i=0;i<5;++i)info.d.push_back(n[d+i]);
      info.p={info.k[0],info.k[1],info.k[2],0,info.k[3],info.k[4],info.k[5],0,info.k[6],info.k[7],info.k[8],0};
    }
    sub_=create_subscription<std_msgs::msg::UInt8MultiArray>("/satellite/stereo_bytes",128,[this](std_msgs::msg::UInt8MultiArray::ConstSharedPtr m){receive(m->data);});
  }
 private:
  void receive(const std::vector<uint8_t>& b){
    if(b.size()<32 || b[0]!='S'||b[1]!='T'||b[2]!='R'||b[3]!='1')return;
    uint32_t id=read32(b,4),part=read32(b,8),count=read32(b,12),w=read32(b,16),h=read32(b,20);
    constexpr size_t chunk=32768;
    if(w==0||h==0||w>4096||h>4096||count!=(4ULL*w*h+chunk-1)/chunk||part>=count)return;
    size_t total=4ULL*w*h,offset=size_t(part)*chunk;
    if(b.size()!=32+std::min(chunk,total-offset))return;
    if(part==0){
      if(next_ && next_!=expected_) RCLCPP_WARN_THROTTLE(get_logger(),*get_clock(),5000,"Dropped incomplete stereo frame %u: %u/%u packets",frame_,next_,expected_);
      frame_=id;next_=0;expected_=count;width_=w;height_=h;data_.assign(total,0);stamp_.sec=read32(b,24);stamp_.nanosec=read32(b,28);
    }
    if(id!=frame_||part!=next_||w!=width_||h!=height_)return;
    std::copy(b.begin()+32,b.end(),data_.begin()+offset);++next_;
    if(next_!=count)return;
    ++completed_;
    if(completed_==1 || completed_%30==0) RCLCPP_INFO(get_logger(),"Reconstructed stereo frame %u (total %u)",id,completed_);
    for(int eye=0;eye<2;++eye){
      sensor_msgs::msg::Image image;image.header.stamp=stamp_;image.header.frame_id=info_[eye].header.frame_id;
      image.width=w;image.height=h;image.encoding="bgr8";image.step=w*3;image.data.resize(size_t(w)*h*3);
      for(size_t i=0;i<size_t(w)*h;++i){size_t src=(eye*size_t(w)*h+i)*2;uint16_t rgb=data_[src]|(uint16_t(data_[src+1])<<8);
        if(rgb==0&&!background_.empty()) {
          int x=int((i%w+eye*background_disparity_)*background_width_/w)%background_width_;
          int y=int((i/w)*background_height_/h);
          size_t bi=(size_t(y)*background_width_+x)*3;
          image.data[i*3]=background_[bi+2];image.data[i*3+1]=background_[bi+1];image.data[i*3+2]=background_[bi];
        } else {
          image.data[i*3]=uint8_t((rgb&31)*255/31);image.data[i*3+1]=uint8_t(((rgb>>5)&63)*255/63);image.data[i*3+2]=uint8_t(((rgb>>11)*255/31));
        }}
      images_[eye]->publish(image);auto info=info_[eye];info.header.stamp=stamp_;info.width=w;info.height=h;infos_[eye]->publish(info);
    }
  }
  uint32_t frame_=0,next_=0,expected_=0,width_=0,height_=0,completed_=0;std::vector<uint8_t> data_;builtin_interfaces::msg::Time stamp_;
  std::vector<uint8_t> background_;int background_width_=0,background_height_=0,background_disparity_=8;
  sensor_msgs::msg::CameraInfo info_[2];rclcpp::Subscription<std_msgs::msg::UInt8MultiArray>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr images_[2];rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr infos_[2];
};
int main(int argc,char** argv){rclcpp::init(argc,argv);try{rclcpp::spin(std::make_shared<Receiver>());}catch(const std::exception& e){fprintf(stderr,"stereo receiver: %s\n",e.what());rclcpp::shutdown();return 1;}rclcpp::shutdown();return 0;}
