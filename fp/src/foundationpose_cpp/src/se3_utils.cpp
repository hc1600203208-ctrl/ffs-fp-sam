#include "se3_utils.h"

#include <algorithm>
#include <cmath>

namespace foundationpose_filter
{

namespace
{

constexpr float kEps = 1e-6F;
constexpr float kPi = 3.14159265358979323846F;

} // namespace

Eigen::Matrix3f skew(const Eigen::Vector3f &v)
{
  Eigen::Matrix3f m;
  m << 0.0F, -v.z(), v.y(),
       v.z(), 0.0F, -v.x(),
      -v.y(), v.x(), 0.0F;
  return m;
}

Eigen::Matrix3f so3Exp(const Eigen::Vector3f &w)
{
  const float theta = w.norm();
  const Eigen::Matrix3f W = skew(w);
  const Eigen::Matrix3f W2 = W * W;
  const Eigen::Matrix3f I = Eigen::Matrix3f::Identity();

  if (theta < kEps)
  {
    return I + W + 0.5F * W2;
  }

  const float theta2 = theta * theta;
  return I + (std::sin(theta) / theta) * W +
         ((1.0F - std::cos(theta)) / theta2) * W2;
}

Eigen::Vector3f so3Log(const Eigen::Matrix3f &R)
{
  const float cos_theta =
      std::clamp((R.trace() - 1.0F) * 0.5F, -1.0F, 1.0F);
  const float theta = std::acos(cos_theta);

  Eigen::Vector3f vee;
  vee << R(2, 1) - R(1, 2),
         R(0, 2) - R(2, 0),
         R(1, 0) - R(0, 1);

  if (theta < kEps)
  {
    return 0.5F * vee;
  }

  if (kPi - theta < 1e-3F)
  {
    Eigen::AngleAxisf aa(R);
    Eigen::Vector3f axis = aa.axis();
    if (!axis.allFinite() || axis.norm() < kEps)
    {
      axis = Eigen::Vector3f::UnitX();
    }
    return theta * axis.normalized();
  }

  return (theta / (2.0F * std::sin(theta))) * vee;
}

Eigen::Matrix4f se3Exp(const Vector6f &xi)
{
  const Eigen::Vector3f v = xi.head<3>();
  const Eigen::Vector3f w = xi.tail<3>();
  const float theta = w.norm();
  const Eigen::Matrix3f W = skew(w);
  const Eigen::Matrix3f W2 = W * W;
  const Eigen::Matrix3f I = Eigen::Matrix3f::Identity();

  Eigen::Matrix3f V;
  if (theta < kEps)
  {
    V = I + 0.5F * W + (1.0F / 6.0F) * W2;
  }
  else
  {
    const float theta2 = theta * theta;
    const float theta3 = theta2 * theta;
    V = I + ((1.0F - std::cos(theta)) / theta2) * W +
        ((theta - std::sin(theta)) / theta3) * W2;
  }

  Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
  T.block<3, 3>(0, 0) = so3Exp(w);
  T.block<3, 1>(0, 3) = V * v;
  return T;
}

Vector6f se3Log(const Eigen::Matrix4f &T)
{
  Eigen::Matrix4f T_orth = T;
  orthonormalizeSE3(&T_orth);

  const Eigen::Matrix3f R = T_orth.block<3, 3>(0, 0);
  const Eigen::Vector3f t = T_orth.block<3, 1>(0, 3);
  const Eigen::Vector3f w = so3Log(R);
  const float theta = w.norm();
  const Eigen::Matrix3f W = skew(w);
  const Eigen::Matrix3f W2 = W * W;
  const Eigen::Matrix3f I = Eigen::Matrix3f::Identity();

  Eigen::Matrix3f V_inv;
  if (theta < kEps)
  {
    V_inv = I - 0.5F * W + (1.0F / 12.0F) * W2;
  }
  else
  {
    const float theta2 = theta * theta;
    const float half_theta = 0.5F * theta;
    const float cot_half_theta = 1.0F / std::tan(half_theta);
    V_inv = I - 0.5F * W +
            ((1.0F / theta2) * (1.0F - half_theta * cot_half_theta)) * W2;
  }

  Vector6f xi;
  xi.head<3>() = V_inv * t;
  xi.tail<3>() = w;
  return xi;
}

Eigen::Matrix4f inverseSE3(const Eigen::Matrix4f &T)
{
  Eigen::Matrix4f T_orth = T;
  orthonormalizeSE3(&T_orth);

  const Eigen::Matrix3f R = T_orth.block<3, 3>(0, 0);
  const Eigen::Vector3f t = T_orth.block<3, 1>(0, 3);

  Eigen::Matrix4f inv = Eigen::Matrix4f::Identity();
  inv.block<3, 3>(0, 0) = R.transpose();
  inv.block<3, 1>(0, 3) = -R.transpose() * t;
  return inv;
}

void orthonormalizeSE3(Eigen::Matrix4f *T)
{
  if (T == nullptr)
  {
    return;
  }

  Eigen::Matrix3f R = T->block<3, 3>(0, 0);
  Eigen::JacobiSVD<Eigen::Matrix3f> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3f U = svd.matrixU();
  Eigen::Matrix3f V = svd.matrixV();
  Eigen::Matrix3f R_orth = U * V.transpose();
  if (R_orth.determinant() < 0.0F)
  {
    U.col(2) *= -1.0F;
    R_orth = U * V.transpose();
  }

  T->block<3, 3>(0, 0) = R_orth;
  T->row(3) << 0.0F, 0.0F, 0.0F, 1.0F;
}

} // namespace foundationpose_filter
