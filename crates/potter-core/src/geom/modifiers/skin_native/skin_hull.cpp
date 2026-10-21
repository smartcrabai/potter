// SPDX-License-Identifier: GPL-3.0-or-later
// Blender hull behavior: source/blender/bmesh/operators/bmo_hull.cc at
// Blender 5.2.2 commit d13f752e3b9c4f8c261cda552b1021f8bcc0382c.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <vector>

#include "btConvexHullComputer.h"

struct PotterSkinHull {
  std::vector<std::uint32_t> triangle_indices;
};

enum PotterSkinHullStatus {
  POTTER_SKIN_HULL_OK = 0,
  POTTER_SKIN_HULL_INVALID_ARGUMENT = 1,
  POTTER_SKIN_HULL_ALLOCATION_FAILED = 2,
  POTTER_SKIN_HULL_INVALID_BULLET_OUTPUT = 3,
};

extern "C" int potter_skin_hull_compute(const float *coordinates,
                                        std::size_t vertex_count,
                                        PotterSkinHull **result)
{
  if (result == nullptr) {
    return POTTER_SKIN_HULL_INVALID_ARGUMENT;
  }
  *result = nullptr;
  if (coordinates == nullptr || vertex_count < 3 ||
      vertex_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
  {
    return POTTER_SKIN_HULL_INVALID_ARGUMENT;
  }

  try {
    std::unique_ptr<btConvexHullComputer> hull(new btConvexHullComputer());
    hull->compute(coordinates, sizeof(float) * 3, static_cast<int>(vertex_count), 0, 0);

    std::unique_ptr<PotterSkinHull> output(new PotterSkinHull());
    std::vector<std::uint32_t> face_vertices;
    for (int face_index = 0; face_index < hull->faces.size(); ++face_index) {
      const int first_edge = hull->faces[face_index];
      if (first_edge < 0 || first_edge >= hull->edges.size()) {
        return POTTER_SKIN_HULL_INVALID_BULLET_OUTPUT;
      }
      const btConvexHullComputer::Edge *first = &hull->edges[first_edge];
      const btConvexHullComputer::Edge *edge = first;
      face_vertices.clear();
      do {
        const int hull_vertex = edge->getTargetVertex();
        if (hull_vertex < 0 || hull_vertex >= hull->original_vertex_index.size()) {
          return POTTER_SKIN_HULL_INVALID_BULLET_OUTPUT;
        }
        const int original_vertex = hull->original_vertex_index[hull_vertex];
        if (original_vertex < 0 || static_cast<std::size_t>(original_vertex) >= vertex_count) {
          return POTTER_SKIN_HULL_INVALID_BULLET_OUTPUT;
        }
        face_vertices.push_back(static_cast<std::uint32_t>(original_vertex));
        edge = edge->getNextEdgeOfFace();
        if (face_vertices.size() > static_cast<std::size_t>(hull->edges.size())) {
          return POTTER_SKIN_HULL_INVALID_BULLET_OUTPUT;
        }
      } while (edge != first);

      for (std::size_t corner = 2; corner < face_vertices.size(); ++corner) {
        output->triangle_indices.push_back(face_vertices[0]);
        output->triangle_indices.push_back(face_vertices[corner - 1]);
        output->triangle_indices.push_back(face_vertices[corner]);
      }
    }

    *result = output.release();
    return POTTER_SKIN_HULL_OK;
  }
  catch (const std::bad_alloc &) {
    return POTTER_SKIN_HULL_ALLOCATION_FAILED;
  }
  catch (...) {
    return POTTER_SKIN_HULL_INVALID_BULLET_OUTPUT;
  }
}

extern "C" std::size_t potter_skin_hull_triangle_index_count(const PotterSkinHull *hull)
{
  return hull == nullptr ? 0 : hull->triangle_indices.size();
}

extern "C" int potter_skin_hull_copy_triangle_indices(const PotterSkinHull *hull,
                                                      std::uint32_t *destination,
                                                      std::size_t capacity)
{
  if (hull == nullptr || (destination == nullptr && !hull->triangle_indices.empty())) {
    return POTTER_SKIN_HULL_INVALID_ARGUMENT;
  }
  if (capacity < hull->triangle_indices.size()) {
    return POTTER_SKIN_HULL_INVALID_ARGUMENT;
  }
  if (!hull->triangle_indices.empty()) {
    std::memcpy(destination,
                hull->triangle_indices.data(),
                hull->triangle_indices.size() * sizeof(std::uint32_t));
  }
  return POTTER_SKIN_HULL_OK;
}

extern "C" void potter_skin_hull_free(PotterSkinHull *hull)
{
  delete hull;
}
