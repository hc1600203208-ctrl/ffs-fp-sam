#pragma once

#include <vector>
#include <Eigen/Dense>
#include <opencv2/core.hpp>

namespace detection_6d {

/**
 * @brief Abstract base class for mesh loading and data access interfaces
 *
 * @note Implementations should handle different mesh formats while maintaining
 *       consistent vertex/face data organization
 */
class BaseMeshLoader {
public:
  virtual ~BaseMeshLoader() = default;

  using Vector3ui = Eigen::Matrix<uint32_t, 3, 1>;

  /**
   * @brief Get identifier name for the loaded mesh
   * @return Mesh name string
   */
  virtual std::string GetName() const noexcept = 0;

  /** @brief Get diameter of the mesh's bounding sphere */
  virtual float GetMeshDiameter() const noexcept = 0;

  /** @brief Get number of vertices in the mesh */
  virtual size_t GetMeshNumVertices() const noexcept = 0;

  /** @brief Get number of triangular faces in the mesh */
  virtual size_t GetMeshNumFaces() const noexcept = 0;

  /** @brief Access array of vertex positions (3D coordinates) */
  virtual const std::vector<Eigen::Vector3f> &GetMeshVertices() const noexcept = 0;

  /** @brief Access array of vertex normals */
  virtual const std::vector<Eigen::Vector3f> &GetMeshVertexNormals() const noexcept = 0;

  /** @brief Access array of texture coordinates (UV mapping) */
  virtual const std::vector<Eigen::Vector3f> &GetMeshTextureCoords() const noexcept = 0;

  /** @brief Access array of triangular face indices */
  virtual const std::vector<Vector3ui> &GetMeshTriangleFaces() const noexcept = 0;

  /** @brief Get centroid position of the mesh model */
  virtual const Eigen::Vector3f &GetMeshModelCenter() const noexcept = 0;

  /**
   * @brief Get oriented bounding box transform
   * @return 4x4 transform from oriented bounding box coordinates to original mesh coordinates
   */
  virtual const Eigen::Matrix4f &GetOrientBounds() const noexcept = 0;

  /** @brief Get dimensional measurements of the object (width/height/depth) */
  virtual const Eigen::Vector3f &GetObjectDimension() const noexcept = 0;

  /** @brief Access texture map image for the mesh */
  virtual const cv::Mat &GetTextureMap() const noexcept = 0;
};

/**
 * @brief Convert pose from original mesh coordinate frame to centered mesh coordinate frame
 *
 * FoundationPose renders a copy of the mesh whose vertices are shifted by -model_center.
 * Python FoundationPose keeps that pose internally, but returns pose @ T(-model_center).
 * This helper performs the inverse conversion for callers that provide the returned,
 * original-mesh pose as a tracking hypothesis.
 *
 * @param pose_in_mesh Input pose in original mesh coordinate system
 * @param mesh_loader Mesh loader containing transformation parameters
 * @return Pose in centered mesh coordinate system used internally by the renderer/refiner
 */
inline Eigen::Matrix4f ConvertPoseMesh2CenteredMesh(
    const Eigen::Matrix4f                 &pose_in_mesh,
    const std::shared_ptr<BaseMeshLoader> &mesh_loader)
{
  Eigen::Matrix4f tf_to_centered_mesh   = Eigen::Matrix4f::Identity();
  tf_to_centered_mesh.block<3, 1>(0, 3) = mesh_loader->GetMeshModelCenter();
  return pose_in_mesh * tf_to_centered_mesh;
}

/**
 * @brief Convert pose from centered mesh coordinate frame to original mesh coordinate frame
 *
 * @param pose_in_centered_mesh Input pose in centered mesh coordinate system
 * @param mesh_loader Mesh loader containing transformation parameters
 * @return Pose in original mesh coordinate system, matching official Python FoundationPose output
 */
inline Eigen::Matrix4f ConvertPoseCenteredMesh2Mesh(
    const Eigen::Matrix4f                 &pose_in_centered_mesh,
    const std::shared_ptr<BaseMeshLoader> &mesh_loader)
{
  Eigen::Matrix4f tf_to_mesh   = Eigen::Matrix4f::Identity();
  tf_to_mesh.block<3, 1>(0, 3) = -mesh_loader->GetMeshModelCenter();
  return pose_in_centered_mesh * tf_to_mesh;
}

/**
 * @brief Convert pose from original mesh coordinate frame to bounding box frame
 *
 * @param pose_in_mesh Input pose in original mesh coordinate system
 * @param mesh_loader Mesh loader containing transformation parameters
 * @return Transformed pose in bounding box coordinate system
 *
 * @note Transformation formula:
 *       T_bbox = T_mesh * T_orient
 *       Where T_orient maps the oriented bounding box coordinates to original mesh coordinates.
 */
inline Eigen::Matrix4f ConvertPoseMesh2BBox(const Eigen::Matrix4f                 &pose_in_mesh,
                                            const std::shared_ptr<BaseMeshLoader> &mesh_loader)
{
  return pose_in_mesh * mesh_loader->GetOrientBounds();
}

/**
 * @brief Factory function for creating Assimp-based mesh loader
 *
 * @param name Identifier for the mesh
 * @param mesh_file_path Path to mesh file (supports .obj/.ply/.stl etc.)
 * @return Shared pointer to initialized mesh loader instance
 *
 * @note Throws std::runtime_error if mesh loading fails
 */
std::shared_ptr<BaseMeshLoader> CreateAssimpMeshLoader(const std::string &name,
                                                       const std::string &mesh_file_path);

} // namespace detection_6d
