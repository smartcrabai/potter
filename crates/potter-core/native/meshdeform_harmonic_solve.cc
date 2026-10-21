// SPDX-License-Identifier: GPL-3.0-or-later

// Mirrors Blender 5.2.2 intern/eigen/linear_solver.cc (EIG_linear_solver_solve, Eigen SparseLU path,
// lines 290-328) for the Mesh Deform harmonic bind (meshlaplacian.cc:1445-1571). The system is an
// Eigen::SparseMatrix<double, ColMajor> factored with Eigen::SparseLU and default ordering, as in
// Blender. Eigen commit 8a1083e9bf41b91fdea6546681f806154efdc25a.
// Eigen is MPL-2.0; see vendor/eigen/COPYING.MPL2.
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

#include <cstddef>
#include <limits>
#include <memory>
#include <vector>

namespace {

using SparseMatrix = Eigen::SparseMatrix<double, Eigen::ColMajor>;
using Triplet = Eigen::Triplet<double>;

// Status codes returned to Rust. Zero is success; every other value leaves the outputs unspecified.
constexpr int kStatusOk = 0;
constexpr int kStatusInvalidArgument = 1;
constexpr int kStatusFactorizationFailed = 2;
constexpr int kStatusSolveFailed = 3;
constexpr int kStatusException = 4;

// The factorization of one bind system. Rust owns it through an opaque pointer.
struct HarmonicFactor {
  Eigen::Index size = 0;
  Eigen::SparseLU<SparseMatrix> solver;
};

}  // namespace

// Factors the `size` x `size` system given as triplets (`rows[i]`, `cols[i]`, `values[i]`) and returns
// the factorization in `*factor`. Rust releases it with potter_meshdeform_harmonic_free.
//
// The caller guarantees that `rows`, `cols`, and `values` hold `entry_count` elements each. Indices are
// checked here; sizes that Eigen cannot represent are rejected before any allocation. The triplets are
// copied into the factorization, so the caller's arrays need not outlive this call.
extern "C" int potter_meshdeform_harmonic_factor(
    std::size_t size,
    std::size_t entry_count,
    const std::size_t *rows,
    const std::size_t *cols,
    const double *values,
    void **factor)
{
  if (factor == nullptr) {
    return kStatusInvalidArgument;
  }
  *factor = nullptr;
  if (size == 0 || size > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      entry_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return kStatusInvalidArgument;
  }
  if (entry_count > 0 && (rows == nullptr || cols == nullptr || values == nullptr)) {
    return kStatusInvalidArgument;
  }

  try {
    std::vector<Triplet> triplets;
    triplets.reserve(entry_count);
    for (std::size_t entry = 0; entry < entry_count; ++entry) {
      if (rows[entry] >= size || cols[entry] >= size) {
        return kStatusInvalidArgument;
      }
      triplets.emplace_back(static_cast<int>(rows[entry]),
                            static_cast<int>(cols[entry]),
                            values[entry]);
    }

    SparseMatrix matrix(static_cast<int>(size), static_cast<int>(size));
    matrix.setFromTriplets(triplets.begin(), triplets.end());
    matrix.makeCompressed();

    std::unique_ptr<HarmonicFactor> owned(new HarmonicFactor);
    owned->size = static_cast<Eigen::Index>(size);
    // Blender: sparseLU->compute(M); result = (info() == Success).
    owned->solver.compute(matrix);
    if (owned->solver.info() != Eigen::Success) {
      return kStatusFactorizationFailed;
    }
    *factor = owned.release();
    return kStatusOk;
  } catch (...) {
    // Allocation failure inside Eigen must not unwind across the C ABI.
    return kStatusException;
  }
}

// Solves one right-hand side against a factorization from potter_meshdeform_harmonic_factor. `rhs` and
// `solution` each hold the factorization's `size` values. Blender solves each cage vertex's right-hand
// side against the same factorization, so solving column by column matches its per-column result.
extern "C" int potter_meshdeform_harmonic_solve_column(const void *factor,
                                                       const double *rhs,
                                                       double *solution)
{
  if (factor == nullptr || rhs == nullptr || solution == nullptr) {
    return kStatusInvalidArgument;
  }
  const auto *harmonic = static_cast<const HarmonicFactor *>(factor);
  try {
    const Eigen::Map<const Eigen::VectorXd> right(rhs, harmonic->size);
    const Eigen::VectorXd answer = harmonic->solver.solve(right);
    if (harmonic->solver.info() != Eigen::Success) {
      return kStatusSolveFailed;
    }
    Eigen::Map<Eigen::VectorXd>(solution, harmonic->size) = answer;
    return kStatusOk;
  } catch (...) {
    return kStatusException;
  }
}

// Releases a factorization returned by potter_meshdeform_harmonic_factor. Null is ignored.
extern "C" void potter_meshdeform_harmonic_free(void *factor)
{
  delete static_cast<HarmonicFactor *>(factor);
}
