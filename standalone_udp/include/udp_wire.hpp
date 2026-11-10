#pragma once

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <zlib.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace udp_wire {
constexpr std::size_t header_size = 32;
constexpr std::size_t chunk_size = 1152;
constexpr std::size_t max_message_size = 8 * 1024 * 1024;
enum Kind : std::uint8_t { control = 1, ack = 2, stereo_png = 3, truth = 4, result = 5 };

inline void put16(std::vector<std::uint8_t>& b, std::uint16_t v) {
  b.push_back(v & 255); b.push_back(v >> 8);
}
inline void put32(std::vector<std::uint8_t>& b, std::uint32_t v) {
  for (int i = 0; i < 4; ++i) b.push_back((v >> (8 * i)) & 255);
}
inline void put64(std::vector<std::uint8_t>& b, std::uint64_t v) {
  put32(b, static_cast<std::uint32_t>(v)); put32(b, static_cast<std::uint32_t>(v >> 32));
}
inline std::uint16_t get16(const std::uint8_t* b) { return b[0] | (std::uint16_t(b[1]) << 8); }
inline std::uint32_t get32(const std::uint8_t* b) {
  return get16(b) | (std::uint32_t(get16(b + 2)) << 16);
}
inline std::uint64_t get64(const std::uint8_t* b) { return get32(b) | (std::uint64_t(get32(b + 4)) << 32); }

struct Message {
  Kind kind;
  std::uint32_t id;
  std::uint64_t stamp_ns;
  std::vector<std::uint8_t> payload;
};

inline std::vector<std::vector<std::uint8_t>> encode(const Message& message) {
  if (message.payload.size() > max_message_size) throw std::invalid_argument("UDP message too large");
  std::size_t count = std::max<std::size_t>(1, (message.payload.size() + chunk_size - 1) / chunk_size);
  if (count > 65535) throw std::invalid_argument("too many UDP chunks");
  std::vector<std::vector<std::uint8_t>> chunks;
  chunks.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    std::size_t begin = i * chunk_size;
    std::size_t n = std::min(chunk_size, message.payload.size() - begin);
    std::vector<std::uint8_t> b{'S', 'U', 'D', '1', 1, static_cast<std::uint8_t>(message.kind), 0, 0};
    put32(b, message.id); put16(b, i); put16(b, count);
    put32(b, message.payload.size()); put64(b, message.stamp_ns);
    auto checksum = crc32(0, message.payload.data() + begin, n);
    put32(b, checksum);
    b.insert(b.end(), message.payload.begin() + begin, message.payload.begin() + begin + n);
    chunks.push_back(std::move(b));
  }
  return chunks;
}

inline void send(int fd, const sockaddr_in& peer, const Message& message) {
  for (const auto& chunk : encode(message)) {
    auto sent = sendto(fd, chunk.data(), chunk.size(), 0,
                       reinterpret_cast<const sockaddr*>(&peer), sizeof(peer));
    if (sent != static_cast<ssize_t>(chunk.size())) throw std::runtime_error("UDP sendto failed");
  }
}

class Reassembler {
  struct Pending {
    std::uint64_t stamp_ns;
    std::uint32_t total;
    std::vector<std::vector<std::uint8_t>> chunks;
    std::size_t received{0};
    std::chrono::steady_clock::time_point created{std::chrono::steady_clock::now()};
  };
  std::map<std::tuple<std::uint8_t, std::uint32_t, std::uint32_t, std::uint16_t>, Pending> pending_;
public:
  bool push(const std::uint8_t* b, std::size_t size, std::uint32_t peer_ip,
            std::uint16_t peer_port, Message* output) {
    if (size < header_size || std::memcmp(b, "SUD1", 4) != 0 || b[4] != 1 ||
        b[5] < control || b[5] > result || b[6] || b[7]) return false;
    auto id = get32(b + 8);
    auto part = get16(b + 12);
    auto count = get16(b + 14);
    auto total = get32(b + 16);
    auto stamp = get64(b + 20);
    if (!count || part >= count || total > max_message_size ||
        count != std::max<std::size_t>(1, (std::size_t(total) + chunk_size - 1) / chunk_size)) return false;
    auto expected = std::min<std::size_t>(chunk_size, std::size_t(total) - std::size_t(part) * chunk_size);
    if (size != header_size + expected || get32(b + 28) != crc32(0, b + header_size, expected)) return false;
    auto key = std::make_tuple(b[5], id, peer_ip, peer_port);
    auto it = pending_.find(key);
    if (it == pending_.end() || it->second.total != total || it->second.stamp_ns != stamp ||
        it->second.chunks.size() != count) {
      it = pending_.insert_or_assign(key, Pending{stamp, total, std::vector<std::vector<std::uint8_t>>(count)}).first;
    }
    auto& p = it->second;
    if (p.chunks[part].empty()) {
      p.chunks[part].assign(b + header_size, b + size);
      ++p.received;
    }
    if (p.received != count) return false;
    output->kind = static_cast<Kind>(b[5]); output->id = id; output->stamp_ns = stamp;
    output->payload.clear(); output->payload.reserve(total);
    for (auto& chunk : p.chunks) output->payload.insert(output->payload.end(), chunk.begin(), chunk.end());
    pending_.erase(it);
    return output->payload.size() == total;
  }

  void expire() {
    auto now = std::chrono::steady_clock::now();
    for (auto it = pending_.begin(); it != pending_.end();) {
      if (now - it->second.created > std::chrono::seconds(3)) it = pending_.erase(it);
      else ++it;
    }
  }
};
}  // namespace udp_wire
