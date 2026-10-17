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
#include <vector>

namespace {

using SparseMatrix = Eigen::SparseMatrix<double, Eigen::ColMajor>;
using Triplet = Eigen::Triplet<double>;

// Status codes returned to Rust. Zero is success; every other value leaves `columns` unspecified.
constexpr int kStatusOk = 0;
constexpr int kStatusInvalidArgument = 1;
constexpr int kStatusFactorizationFailed = 2;
constexpr int kStatusSolveFailed = 3;
constexpr int kStatusException = 4;

}  // namespace

// Factors the `size` x `size` system given as triplets (`rows[i]`, `cols[i]`, `values[i]`) and solves it
// for `column_count` right-hand sides. `columns` holds the right-hand sides as contiguous column-major
// columns of length `size` (one column per cage vertex) and is overwritten with the solutions.
//
// The caller guarantees that `rows`, `cols`, and `values` hold `entry_count` elements each and that
// `columns` holds `size * column_count` elements. Indices are checked here; sizes that Eigen cannot
// represent are rejected before any allocation.
extern "C" int potter_meshdeform_harmonic_solve(
    std::size_t size,
    std::size_t entry_count,
    const std::size_t *rows,
    const std::size_t *cols,
    const double *values,
    std::size_t column_count,
    double *columns)
{
  if (size == 0 || size > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      entry_count > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      column_count > static_cast<std::size_t>(std::numeric_limits<Eigen::Index>::max()) / size) {
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

    // Blender: sparseLU->compute(M); result = (info() == Success).
    Eigen::SparseLU<SparseMatrix> solver;
    solver.compute(matrix);
    if (solver.info() != Eigen::Success) {
      return kStatusFactorizationFailed;
    }

    // Blender solves each right-hand side against the same factorization. A block solve applies the
    // same triangular solves column by column, so each column matches the per-column result.
    Eigen::Map<Eigen::MatrixXd> block(columns,
                                      static_cast<Eigen::Index>(size),
                                      static_cast<Eigen::Index>(column_count));
    const Eigen::MatrixXd solution = solver.solve(block);
    if (solver.info() != Eigen::Success) {
      return kStatusSolveFailed;
    }
    block = solution;
    return kStatusOk;
  } catch (...) {
    // Allocation failure inside Eigen must not unwind across the C ABI.
    return kStatusException;
  }
}
