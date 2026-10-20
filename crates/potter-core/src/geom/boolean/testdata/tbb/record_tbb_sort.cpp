// Records the order `tbb::parallel_sort` gives tied keys, for the Potter model in
// `geom/boolean/libcxx_sort.rs` (Blender's `IMesh::populate_vert` sort in a TBB build).
//
// Build against the oneTBB release Blender 5.2.2 ships (v2022.3.0, `versions.cmake`): the `include/`
// directory of the oneTBB v2022.3.0 source tree. Link Blender's own runtime, then run once:
//
// The same harness built against oneTBB master's headers orders five cases differently:
// populate_shaped_1921, pairs_shuffled_1200, interleaved_runs_600, random_ties_2000, heavy_ties_1000.
// Those cases tell the versions apart; the others are shared coverage.
//
//   clang++ -std=c++17 -O2 -I$TBB_2022_INCLUDE record_tbb_sort.cpp \
//       -L/Applications/Blender.app/Contents/Resources/lib -ltbb \
//       -Wl,-rpath,/Applications/Blender.app/Contents/Resources/lib -o /tmp/record_tbb_sort
//   /tmp/record_tbb_sort > tbb_sort_v2022_3_0.txt
//
// Every case is generated from a fixed mt19937 stream (no `std::shuffle` or `std::uniform_*`, whose
// results differ between standard libraries). Each case is sorted with 1 and with 6 threads, many
// times; the program stops if any run disagrees, because the model is only valid for a fixed order.
//
// Output format, one case per block:
//   case <name> <n>
//   keys <k_0> ... <k_{n-1}>      (the comparator is key[a] < key[b] on element indices)
//   order <i_0> ... <i_{n-1}>     (element indices after `parallel_sort`, from the identity order)

#include <tbb/global_control.h>
#include <tbb/parallel_sort.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

// Portable draw in [0, bound): raw mt19937 output, not a standard-library distribution.
std::uint32_t draw(std::mt19937 &rng, std::uint32_t bound) { return rng() % bound; }

// Fisher-Yates over `values` with the portable draw.
template <typename T> void shuffle_portable(std::vector<T> &values, std::mt19937 &rng) {
  for (std::size_t i = values.size(); i > 1; --i) {
    std::size_t j = draw(rng, static_cast<std::uint32_t>(i));
    std::swap(values[i - 1], values[j]);
  }
}

struct Case {
  std::string name;
  std::vector<std::int64_t> keys;
};

// Populate-shaped keys (`IMesh::populate_vert`): input vertices sort by origin, created vertices sort
// after all inputs by id. Origins tie across operands; created ids are unique.
Case populate_shaped(std::uint32_t seed, std::size_t n) {
  std::mt19937 rng(seed);
  Case c{"populate_shaped_" + std::to_string(n), {}};
  const std::int64_t created_base = std::int64_t{1} << 40;
  std::int64_t next_id = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (draw(rng, 3) == 0) {
      c.keys.push_back(created_base + next_id++);
    } else {
      c.keys.push_back(static_cast<std::int64_t>(draw(rng, static_cast<std::uint32_t>(n / 4))));
    }
  }
  return c;
}

// Pairs of equal keys: every 10th element ties with its predecessor, keys shuffled.
Case pairs_shuffled(std::uint32_t seed, std::size_t n) {
  std::mt19937 rng(seed);
  Case c{"pairs_shuffled_" + std::to_string(n), {}};
  for (std::size_t i = 0; i < n; ++i) {
    c.keys.push_back(static_cast<std::int64_t>(i) - ((i % 10 == 1) ? 1 : 0));
  }
  shuffle_portable(c.keys, rng);
  return c;
}

// Two interleaved runs that share key values.
Case interleaved_runs(std::size_t n) {
  Case c{"interleaved_runs_" + std::to_string(n), {}};
  for (std::size_t i = 0; i < n; ++i) {
    std::int64_t k = (i % 2) ? static_cast<std::int64_t>(i / 2)
                             : static_cast<std::int64_t>(n / 2 - i / 2);
    c.keys.push_back(k);
  }
  return c;
}

// Random keys in [0, n/2): many ties of varying size.
Case random_ties(std::uint32_t seed, std::size_t n) {
  std::mt19937 rng(seed);
  Case c{"random_ties_" + std::to_string(n), {}};
  for (std::size_t i = 0; i < n; ++i) {
    c.keys.push_back(static_cast<std::int64_t>(draw(rng, static_cast<std::uint32_t>(n / 2))));
  }
  return c;
}

// Heavy ties: four distinct values.
Case heavy_ties(std::uint32_t seed, std::size_t n) {
  std::mt19937 rng(seed);
  Case c{"heavy_ties_" + std::to_string(n), {}};
  for (std::size_t i = 0; i < n; ++i) {
    c.keys.push_back(static_cast<std::int64_t>(draw(rng, 4)));
  }
  return c;
}

// Already sorted and reversed inputs exercise the pretest and the all-equal-pivot path.
Case sorted_input(std::size_t n) {
  Case c{"sorted_" + std::to_string(n), {}};
  for (std::size_t i = 0; i < n; ++i) c.keys.push_back(static_cast<std::int64_t>(i));
  return c;
}

Case reversed_input(std::size_t n) {
  Case c{"reversed_" + std::to_string(n), {}};
  for (std::size_t i = 0; i < n; ++i) c.keys.push_back(static_cast<std::int64_t>(n - i));
  return c;
}

std::vector<int> sort_order(const std::vector<std::int64_t> &keys, int threads) {
  tbb::global_control limit(tbb::global_control::max_allowed_parallelism,
                            static_cast<std::size_t>(threads));
  std::vector<int> order(keys.size());
  for (std::size_t i = 0; i < order.size(); ++i) order[i] = static_cast<int>(i);
  tbb::parallel_sort(order.begin(), order.end(),
                     [&keys](int a, int b) { return keys[a] < keys[b]; });
  return order;
}

// Sorts with 1 and 6 threads, repeatedly. Exits if any run differs from the first.
std::vector<int> stable_order(const Case &c) {
  std::vector<int> first = sort_order(c.keys, 1);
  for (int threads : {1, 6}) {
    for (int rep = 0; rep < 10; ++rep) {
      if (sort_order(c.keys, threads) != first) {
        std::fprintf(stderr, "%s: parallel_sort order differs at %d threads (rep %d)\n",
                     c.name.c_str(), threads, rep);
        std::exit(1);
      }
    }
  }
  return first;
}

void print_case(const Case &c, const std::vector<int> &order) {
  std::printf("case %s %zu\n", c.name.c_str(), c.keys.size());
  std::printf("keys");
  for (std::int64_t k : c.keys) std::printf(" %lld", static_cast<long long>(k));
  std::printf("\norder");
  for (int i : order) std::printf(" %d", i);
  std::printf("\n");
}

}  // namespace

int main() {
  std::vector<Case> cases;
  cases.push_back(populate_shaped(11, 1921));
  cases.push_back(populate_shaped(12, 600));
  cases.push_back(pairs_shuffled(21, 500));
  cases.push_back(pairs_shuffled(22, 1200));
  cases.push_back(interleaved_runs(600));
  cases.push_back(random_ties(31, 2000));
  cases.push_back(heavy_ties(41, 1000));
  cases.push_back(sorted_input(700));
  cases.push_back(reversed_input(700));
  std::printf("# oneTBB v2022.3.0 tbb::parallel_sort recordings (see record_tbb_sort.cpp)\n");
  for (const Case &c : cases) print_case(c, stable_order(c));
  return 0;
}
