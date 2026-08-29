#include "grounded_sam/image_utils.hpp"
#include "grounded_sam/pipeline.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct CliArgs {
  std::string input;
  std::string output_dir{"outputs"};
  std::string dino_engine{"engines/grounding_dino_fixed_prompt.engine"};
  std::string sam_encoder_engine{"engines/sam_image_encoder.engine"};
  std::string sam_decoder_engine{"engines/sam_mask_decoder.engine"};
  float box_threshold{0.3f};
  int max_detections{16};
};

void printUsage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0 << " --input <image_or_folder> --output_dir <dir> [options]\n"
      << "\nOptions:\n"
      << "  --dino_engine <path>\n"
      << "  --sam_encoder_engine <path>\n"
      << "  --sam_decoder_engine <path>\n"
      << "  --box_threshold <float>   default 0.3\n"
      << "  --max_detections <int>    default 16\n";
}

CliArgs parseArgs(int argc, char** argv) {
  CliArgs args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    auto require_value = [&](const std::string& name) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("Missing value for " + name);
      return argv[++i];
    };

    if (key == "--input") {
      args.input = require_value(key);
    } else if (key == "--output_dir") {
      args.output_dir = require_value(key);
    } else if (key == "--dino_engine") {
      args.dino_engine = require_value(key);
    } else if (key == "--sam_encoder_engine") {
      args.sam_encoder_engine = require_value(key);
    } else if (key == "--sam_decoder_engine") {
      args.sam_decoder_engine = require_value(key);
    } else if (key == "--box_threshold") {
      args.box_threshold = std::stof(require_value(key));
    } else if (key == "--max_detections") {
      args.max_detections = std::stoi(require_value(key));
    } else if (key == "--help" || key == "-h") {
      printUsage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + key);
    }
  }
  if (args.input.empty()) {
    throw std::runtime_error("--input is required");
  }
  return args;
}

std::string outputMaskPath(const std::string& output_dir, const std::string& image_path) {
  namespace fs = std::filesystem;
  const fs::path stem = fs::path(image_path).stem();
  return (fs::path(output_dir) / (stem.string() + "_mask.png")).string();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const CliArgs cli = parseArgs(argc, argv);
    std::filesystem::create_directories(cli.output_dir);

    grounded_sam::PipelineOptions options;
    options.dino_engine = cli.dino_engine;
    options.sam_encoder_engine = cli.sam_encoder_engine;
    options.sam_decoder_engine = cli.sam_decoder_engine;
    options.box_threshold = cli.box_threshold;
    options.max_detections = cli.max_detections;

    grounded_sam::GroundedSamPipeline pipeline(options);
    const auto images = grounded_sam::listInputImages(cli.input);
    if (images.empty()) {
      std::cerr << "No input images found.\n";
      return 1;
    }

    for (const auto& image_path : images) {
      std::cerr << "Processing " << image_path << '\n';
      const cv::Mat rgb = grounded_sam::readRgbImage(image_path);
      const cv::Mat mask = pipeline.run(rgb);
      const std::string out_path = outputMaskPath(cli.output_dir, image_path);
      grounded_sam::saveMaskPng(out_path, mask);
      std::cerr << "Saved " << out_path << " positive_pixels=" << cv::countNonZero(mask) << '\n';
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
    printUsage(argv[0]);
    return 1;
  }
}

