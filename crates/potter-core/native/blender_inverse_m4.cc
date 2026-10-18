// SPDX-License-Identifier: GPL-3.0-or-later

// Mirrors Blender 5.2.2 intern/eigen/intern/matrix.cc using Eigen commit
// 8a1083e9bf41b91fdea6546681f806154efdc25a (Eigen/src/LU/InverseImpl.h).
// Upstream archive SHA-256: cc28a84fdec496c6777596350ea805519bf10f717d21044ae6ba3dd562183a26.
// Eigen is MPL-2.0; see vendor/eigen/COPYING.MPL2.
#include <Eigen/LU>

#include <cstring>

extern "C" bool potter_blender_inverse_m4(const float matrix[16], float inverse[16])
{
  Eigen::Map<const Eigen::Matrix4f> input(matrix);
  Eigen::Matrix4f result;
  bool invertible = true;
  input.computeInverseWithCheck(result, invertible, 0.0f);
  if (invertible) {
    std::memcpy(inverse, result.data(), sizeof(float) * 16);
  }
  return invertible;
}
