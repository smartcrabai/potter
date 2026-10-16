// SPDX-License-Identifier: GPL-3.0-or-later

// Mirrors Blender 5.2.2 intern/eigen/intern/matrix.cc using Eigen commit
// 8a1083e9bf41b91fdea6546681f806154efdc25a (Eigen/src/LU/InverseImpl.h).
#include <Eigen/LU>

#include <cstring>

extern "C" void potter_blender_inverse_m4(const float matrix[16], float inverse[16])
{
  Eigen::Map<const Eigen::Matrix4f> input(matrix);
  Eigen::Matrix4f result;
  bool invertible = true;
  input.computeInverseWithCheck(result, invertible, 0.0f);
  if (!invertible) {
    result.setZero();
  }
  std::memcpy(inverse, result.data(), sizeof(float) * 16);
}
