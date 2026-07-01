#pragma once

#include <Eigen/Dense>

namespace foundationpose_filter
{

using Vector6f = Eigen::Matrix<float, 6, 1>;

Eigen::Matrix3f skew(const Eigen::Vector3f &v);
Eigen::Matrix3f so3Exp(const Eigen::Vector3f &w);
Eigen::Vector3f so3Log(const Eigen::Matrix3f &R);
Eigen::Matrix4f se3Exp(const Vector6f &xi);
Vector6f se3Log(const Eigen::Matrix4f &T);
Eigen::Matrix4f inverseSE3(const Eigen::Matrix4f &T);
void orthonormalizeSE3(Eigen::Matrix4f *T);

} // namespace foundationpose_filter
