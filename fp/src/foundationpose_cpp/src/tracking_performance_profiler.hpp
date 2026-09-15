#ifndef FOUNDATIONPOSE_CPP_TRACKING_PERFORMANCE_PROFILER_HPP_
#define FOUNDATIONPOSE_CPP_TRACKING_PERFORMANCE_PROFILER_HPP_

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace foundationpose_profiling
{

inline double MillisecondsBetween(const std::chrono::steady_clock::time_point &start,
                                  const std::chrono::steady_clock::time_point &end)
{
  return std::chrono::duration<double, std::milli>(end - start).count();
}

struct TimingStats
{
  void Add(double value_ms)
  {
    if (value_ms < 0.0)
    {
      value_ms = 0.0;
    }
    ++count;
    sum_ms += value_ms;
    minimum_ms = std::min(minimum_ms, value_ms);
    maximum_ms = std::max(maximum_ms, value_ms);
  }

  void Reset()
  {
    count = 0;
    sum_ms = 0.0;
    minimum_ms = std::numeric_limits<double>::max();
    maximum_ms = 0.0;
  }

  double AverageMs() const
  {
    return count == 0 ? 0.0 : sum_ms / static_cast<double>(count);
  }

  std::uint64_t count{0};
  double sum_ms{0.0};
  double minimum_ms{std::numeric_limits<double>::max()};
  double maximum_ms{0.0};
};

struct DepthTimingBreakdown
{
  double remap_prepare_ms{0.0};
  double ffs_depth_estimation_ms{0.0};
  double depth_validity_filter_ms{0.0};
  double depth_confidence_filter_ms{0.0};
  double depth_restore_align_ms{0.0};
};

struct TrackingFrameTiming
{
  double end_to_end_ms{0.0};
  double ffs_depth_estimation_ms{0.0};
  double depth_validity_filter_ms{0.0};
  double depth_confidence_filter_ms{0.0};
  double depth_restore_align_ms{0.0};
  double foundationpose_refine_ms{0.0};
  double se3_filter_ms{0.0};
  double multi_object_dispatch_ms{0.0};
  double multi_object_sync_total_ms{0.0};
  double worker_scheduling_wait_ms{0.0};
  std::vector<double> per_object_refine_ms;
  std::vector<double> per_object_schedule_wait_ms;
};

class TrackingPerformanceProfiler
{
public:
  void Initialize(bool enabled, std::size_t log_interval_frames, int sync_queue_size,
                  double max_sync_interval_sec,
                  const std::vector<std::string> &object_names)
  {
    enabled_ = enabled;
    log_interval_frames_ = log_interval_frames == 0 ? 30 : log_interval_frames;
    sync_queue_size_ = sync_queue_size;
    max_sync_interval_sec_ = max_sync_interval_sec;
    object_names_ = object_names;
    started_at_ = std::chrono::steady_clock::now();
    Reset();
  }

  bool enabled() const { return enabled_; }

  void RecordDroppedBusy()
  {
    ++dropped_busy_frames_;
  }

  void RecordDroppedTimestamp()
  {
    ++dropped_timestamp_frames_;
  }

  void RecordRegistrationFrame()
  {
    ++registration_frames_;
  }

  void RecordTrackingFrame(const TrackingFrameTiming &timing)
  {
    const auto now = std::chrono::steady_clock::now();
    if (tracking_frames_ == 0)
    {
      first_tracked_at_ = now;
    }
    ++tracking_frames_;
    last_tracked_at_ = now;
    end_to_end_.Add(timing.end_to_end_ms);
    ffs_depth_estimation_.Add(timing.ffs_depth_estimation_ms);
    depth_validity_filter_.Add(timing.depth_validity_filter_ms);
    depth_confidence_filter_.Add(timing.depth_confidence_filter_ms);
    depth_restore_align_.Add(timing.depth_restore_align_ms);
    foundationpose_refine_.Add(timing.foundationpose_refine_ms);
    se3_filter_.Add(timing.se3_filter_ms);
    multi_object_dispatch_.Add(timing.multi_object_dispatch_ms);
    multi_object_sync_total_.Add(timing.multi_object_sync_total_ms);
    worker_scheduling_wait_.Add(timing.worker_scheduling_wait_ms);

    for (std::size_t index = 0; index < per_object_refine_.size(); ++index)
    {
      if (index < timing.per_object_refine_ms.size())
      {
        per_object_refine_[index].Add(timing.per_object_refine_ms[index]);
      }
      if (index < timing.per_object_schedule_wait_ms.size())
      {
        per_object_schedule_wait_[index].Add(timing.per_object_schedule_wait_ms[index]);
      }
    }
  }

  bool ShouldLog() const
  {
    return enabled_ && tracking_frames_ > 0 && tracking_frames_ % log_interval_frames_ == 0;
  }

  std::string FormatPeriodicReport() const
  {
    return FormatReport(false);
  }

  std::string FormatFinalSummary() const
  {
    return FormatReport(true);
  }

private:
  void Reset()
  {
    tracking_frames_ = 0;
    registration_frames_ = 0;
    dropped_busy_frames_ = 0;
    dropped_timestamp_frames_ = 0;

    end_to_end_.Reset();
    ffs_depth_estimation_.Reset();
    depth_validity_filter_.Reset();
    depth_confidence_filter_.Reset();
    depth_restore_align_.Reset();
    foundationpose_refine_.Reset();
    se3_filter_.Reset();
    multi_object_dispatch_.Reset();
    multi_object_sync_total_.Reset();
    worker_scheduling_wait_.Reset();

    per_object_refine_.clear();
    per_object_refine_.resize(object_names_.size());
    per_object_schedule_wait_.clear();
    per_object_schedule_wait_.resize(object_names_.size());
  }

  double CurrentFps() const
  {
    if (tracking_frames_ == 0)
    {
      return 0.0;
    }
    const double elapsed_seconds =
        std::chrono::duration<double>(last_tracked_at_ - first_tracked_at_).count();
    if (elapsed_seconds <= 1e-6)
    {
      return 0.0;
    }
    return static_cast<double>(tracking_frames_) / elapsed_seconds;
  }

  std::string FormatStats(const char *name, const TimingStats &stats) const
  {
    std::ostringstream stream;
    stream << name << "_count=" << stats.count;
    if (stats.count == 0)
    {
      stream << " " << name << "_avg=n/a";
      return stream.str();
    }
    stream << " " << name << "_avg=" << std::fixed << stats.AverageMs() << "ms";
    stream << " " << name << "_min=" << std::fixed << stats.minimum_ms << "ms";
    stream << " " << name << "_max=" << std::fixed << stats.maximum_ms << "ms";
    return stream.str();
  }

  std::string FormatPerObjectStats(const char *name, const std::vector<TimingStats> &stats) const
  {
    std::ostringstream stream;
    stream << name << "_per_object=[";
    for (std::size_t index = 0; index < stats.size(); ++index)
    {
      if (index > 0)
      {
        stream << ", ";
      }
      const std::string &object_name =
          index < object_names_.size() ? object_names_[index] : std::to_string(index);
      stream << object_name << ":avg=" << std::fixed << stats[index].AverageMs() << "ms";
      stream << "/min=" << std::fixed << stats[index].minimum_ms << "ms";
      stream << "/max=" << std::fixed << stats[index].maximum_ms << "ms";
    }
    stream << "]";
    return stream.str();
  }

  std::string FormatReport(bool final_summary) const
  {
    if (!enabled_)
    {
      return "";
    }

    std::ostringstream stream;
    stream << (final_summary ? "PERF_FINAL" : "PERF") << " ";
    stream << "tracked_frames=" << tracking_frames_ << " ";
    stream << "registration_frames=" << registration_frames_ << " ";
    stream << "output_fps=" << std::fixed << CurrentFps() << " ";
    stream << "sync_queue_size=" << sync_queue_size_ << " ";
    stream << "max_sync_interval_sec=" << std::fixed << max_sync_interval_sec_ << " ";
    stream << "dropped_busy_frames=" << dropped_busy_frames_ << " ";
    stream << "dropped_timestamp_frames=" << dropped_timestamp_frames_ << " ";
    stream << FormatStats("end_to_end", end_to_end_) << " ";
    stream << FormatStats("ffs_depth_estimation", ffs_depth_estimation_) << " ";
    stream << FormatStats("depth_validity_filter", depth_validity_filter_) << " ";
    stream << FormatStats("depth_confidence_filter", depth_confidence_filter_) << " ";
    stream << FormatStats("depth_restore_align", depth_restore_align_) << " ";
    stream << FormatStats("foundationpose_refine_total", foundationpose_refine_) << " ";
    stream << FormatPerObjectStats("foundationpose_refine", per_object_refine_) << " ";
    stream << FormatStats("se3_filter", se3_filter_) << " ";
    stream << FormatStats("multi_object_dispatch", multi_object_dispatch_) << " ";
    stream << FormatStats("multi_object_sync_total", multi_object_sync_total_) << " ";
    stream << FormatStats("worker_schedule_wait_total", worker_scheduling_wait_) << " ";
    stream << FormatPerObjectStats("worker_schedule_wait", per_object_schedule_wait_);
    return stream.str();
  }

  bool enabled_{false};
  std::size_t log_interval_frames_{30};
  int sync_queue_size_{0};
  double max_sync_interval_sec_{0.0};
  std::vector<std::string> object_names_;
  std::chrono::steady_clock::time_point started_at_{};

  std::uint64_t tracking_frames_{0};
  std::chrono::steady_clock::time_point first_tracked_at_{};
  std::chrono::steady_clock::time_point last_tracked_at_{};
  std::uint64_t registration_frames_{0};
  std::uint64_t dropped_busy_frames_{0};
  std::uint64_t dropped_timestamp_frames_{0};

  TimingStats end_to_end_;
  TimingStats ffs_depth_estimation_;
  TimingStats depth_validity_filter_;
  TimingStats depth_confidence_filter_;
  TimingStats depth_restore_align_;
  TimingStats foundationpose_refine_;
  TimingStats se3_filter_;
  TimingStats multi_object_dispatch_;
  TimingStats multi_object_sync_total_;
  TimingStats worker_scheduling_wait_;
  std::vector<TimingStats> per_object_refine_;
  std::vector<TimingStats> per_object_schedule_wait_;
};

}  // namespace foundationpose_profiling

#endif  // FOUNDATIONPOSE_CPP_TRACKING_PERFORMANCE_PROFILER_HPP_
