#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace sct {
struct Object {
  std::string name, mesh_path;
  std::array<double, 6> pose;
};
struct Config {
  uint32_t width, height, bytes_per_second;
  double fps;
  std::array<double, 6> amplitudes, frequencies;
  std::string calibration_path;
  std::vector<Object> objects;
};
struct Request { uint8_t type; uint32_t id; Config config{}; };

class Reader {
 public:
  explicit Reader(const std::vector<uint8_t>& data) : data_(data) {}
  uint8_t u8() { need(1); return data_[at_++]; }
  uint16_t u16() { uint16_t value=u8(); return value | uint16_t(u8())<<8; }
  uint32_t u32() { uint32_t value=u16(); return value | uint32_t(u16())<<16; }
  double f64() {
    uint64_t bits=u32(); bits |= uint64_t(u32())<<32;
    double value; std::memcpy(&value,&bits,sizeof(value));
    if(!std::isfinite(value)) throw std::invalid_argument("non-finite floating-point value");
    return value;
  }
  std::string string() {
    const auto length=u16();
    if(length==0 || length>4096) throw std::invalid_argument("invalid string length");
    need(length);
    std::string value(data_.begin()+at_,data_.begin()+at_+length);
    at_+=length;
    if(value.find('\0')!=std::string::npos) throw std::invalid_argument("NUL in string");
    return value;
  }
  size_t remaining() const { return data_.size()-at_; }
 private:
  void need(size_t count) const { if(count>remaining()) throw std::invalid_argument("truncated SCT1 packet"); }
  const std::vector<uint8_t>& data_; size_t at_=0;
};

inline Request parse(const std::vector<uint8_t>& data) {
  if(data.size()<16 || data.size()>1048592 ||
     data[0]!='S'||data[1]!='C'||data[2]!='T'||data[3]!='1')
    throw std::invalid_argument("invalid SCT1 header");
  Reader reader(data);
  for(int i=0;i<4;++i) reader.u8();
  Request request; request.type=reader.u8();
  if(reader.u8()!=0 || reader.u16()!=0) throw std::invalid_argument("unsupported SCT1 flags");
  request.id=reader.u32();
  if(reader.u32()!=reader.remaining()) throw std::invalid_argument("SCT1 payload length mismatch");
  if(request.type==2 || request.type==3) {
    if(reader.remaining()!=0) throw std::invalid_argument("command payload must be empty");
    return request;
  }
  if(request.type!=1) throw std::invalid_argument("unsupported SCT1 command");
  auto& c=request.config;
  c.width=reader.u32(); c.height=reader.u32(); c.fps=reader.f64(); c.bytes_per_second=reader.u32();
  const uint16_t object_count=reader.u16();
  if(reader.u16()!=0) throw std::invalid_argument("nonzero SCT1 reserved field");
  if(c.width==0||c.height==0||c.width>4096||c.height>4096||c.fps<=0||c.fps>1000||c.bytes_per_second==0||
     object_count==0||object_count>32) throw std::invalid_argument("invalid SCT1 rendering parameters");
  for(auto& n:c.amplitudes) n=reader.f64();
  for(auto& n:c.frequencies) n=reader.f64();
  c.calibration_path=reader.string();
  for(uint16_t i=0;i<object_count;++i) {
    Object object; object.name=reader.string();object.mesh_path=reader.string();
    for(auto& n:object.pose) n=reader.f64();
    c.objects.push_back(std::move(object));
  }
  if(reader.remaining()!=0) throw std::invalid_argument("trailing SCT1 data");
  return request;
}

inline void u16(std::vector<uint8_t>& b,uint16_t n) { b.push_back(n&255);b.push_back(n>>8); }
inline void u32(std::vector<uint8_t>& b,uint32_t n) {u16(b,n&65535);u16(b,n>>16);}
inline std::vector<uint8_t> ack(uint32_t id,uint8_t status,const std::string& text) {
  std::vector<uint8_t> b{'S','C','T','1',0x80,0,0,0};
  u32(b,id);u32(b,3+static_cast<uint32_t>(text.size()));
  b.push_back(status);u16(b,static_cast<uint16_t>(text.size()));
  b.insert(b.end(),text.begin(),text.end());return b;
}
}  // namespace sct
