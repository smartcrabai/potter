//! Locating cells by exact sorting around an edge, and nesting of disconnected patch components.
//! Blender `find_cell_for_point_near_edge`, `find_ambient_cell`, `find_good_sorting_edge`,
//! `find_containing_cell`, `closest_on_tri_to_point`, `find_component_containers`,
//! `populate_comp_bbs`, `finish_patch_cell_graph`.

use glam::DVec3;
use num_traits::{One, Signed, Zero};

use super::super::exact::{Rat, Rat3, add3, cross3, dot3, scale3, sgn, sub3};
use super::super::imesh::{Arena, NO_INDEX, Tri};
use super::invariant;
use super::patches::{CellsInfo, PatchesInfo, Side, find_patch_components, merge_cells};
use super::sort::{EXTRA_TRI, TriView, sort_tris_around_edge};
use super::topology::{Edge, Topology, find_flap_vert};
use crate::error::Result;

/// Where the closest point of a triangle to a query point lies (Blender's `r_vert` / `r_edge`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Closest {
    /// On corner `k` of the triangle.
    OnVertex(usize),
    /// On the edge from corner `k` to corner `k + 1`.
    OnEdge(usize),
    /// Strictly inside the triangle.
    Inside,
}

/// Blender `ComponentContainer`: a component that contains a point of another component.
#[derive(Clone, Debug)]
struct ComponentContainer {
    nearest_cell: Option<usize>,
    dist_to_cell: Rat,
}

/// Axis-aligned box of double coordinates, used only to skip component pairs that cannot nest.
#[derive(Clone, Copy, Debug)]
struct BBox {
    min: DVec3,
    max: DVec3,
}

impl BBox {
    fn empty() -> Self {
        Self {
            min: DVec3::splat(f64::INFINITY),
            max: DVec3::splat(f64::NEG_INFINITY),
        }
    }

    fn include(&mut self, p: DVec3) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }

    fn might_intersect(&self, other: &Self) -> bool {
        self.min.cmple(other.max).all() && other.min.cmple(self.max).all()
    }
}

/// `checked` division: `num / den`, or an invariant error for a zero denominator (degenerate
/// triangle).
fn checked_div(num: &Rat, den: &Rat) -> Result<Rat> {
    if den.is_zero() {
        return Err(invariant("zero denominator in exact geometry"));
    }
    Ok(num / den)
}

/// Squared distance between two exact points.
fn dist2(p: &Rat3, q: &Rat3) -> Rat {
    let d = sub3(p, q);
    dot3(&d, &d)
}

/// Blender `closest_on_tri_to_point`: exact closest point of triangle `(a, b, c)` to `p`, as
/// squared distance and location.
fn closest_on_tri_to_point(p: &Rat3, a: &Rat3, b: &Rat3, c: &Rat3) -> Result<(Rat, Closest)> {
    let ab = sub3(b, a);
    let ac = sub3(c, a);
    let ap = sub3(p, a);
    let d1 = dot3(&ab, &ap);
    let d2 = dot3(&ac, &ap);
    if sgn(&d1) <= 0 && sgn(&d2) <= 0 {
        return Ok((dist2(p, a), Closest::OnVertex(0)));
    }
    let bp = sub3(p, b);
    let d3 = dot3(&ab, &bp);
    let d4 = dot3(&ac, &bp);
    if sgn(&d3) >= 0 && d4 <= d3 {
        return Ok((dist2(p, b), Closest::OnVertex(1)));
    }
    let vc = &d1 * &d4 - &d3 * &d2;
    if sgn(&vc) <= 0 && sgn(&d1) >= 0 && sgn(&d3) <= 0 {
        let v = checked_div(&d1, &(&d1 - &d3))?;
        let r = add3(a, &scale3(&v, &ab));
        return Ok((dist2(p, &r), Closest::OnEdge(0)));
    }
    let cp = sub3(p, c);
    let d5 = dot3(&ab, &cp);
    let d6 = dot3(&ac, &cp);
    if sgn(&d6) >= 0 && d5 <= d6 {
        return Ok((dist2(p, c), Closest::OnVertex(2)));
    }
    let vb = &d5 * &d2 - &d1 * &d6;
    if sgn(&vb) <= 0 && sgn(&d2) >= 0 && sgn(&d6) <= 0 {
        let w = checked_div(&d2, &(&d2 - &d6))?;
        let r = add3(a, &scale3(&w, &ac));
        return Ok((dist2(p, &r), Closest::OnEdge(2)));
    }
    let va = &d3 * &d6 - &d5 * &d4;
    let d4_minus_d3 = &d4 - &d3;
    let d5_minus_d6 = &d5 - &d6;
    if sgn(&va) <= 0 && sgn(&d4_minus_d3) >= 0 && sgn(&d5_minus_d6) >= 0 {
        let w = checked_div(&d4_minus_d3, &(&d4_minus_d3 + &d5_minus_d6))?;
        let bc = sub3(c, b);
        let r = add3(b, &scale3(&w, &bc));
        return Ok((dist2(p, &r), Closest::OnEdge(1)));
    }
    let denom = checked_div(&Rat::one(), &(&va + &vb + &vc))?;
    let v = &vb * &denom;
    let w = &vc * &denom;
    let r = add3(&add3(a, &scale3(&v, &ab)), &scale3(&w, &ac));
    Ok((dist2(p, &r), Closest::Inside))
}

/// Blender `find_cell_for_point_near_edge`: the cell containing point `p`, found by inserting a
/// dummy triangle `(e.v0, e.v1, p)` into the sort around `e`.
fn find_cell_for_point_near_edge(
    arena: &mut Arena,
    tris: &[Tri],
    topo: &Topology,
    pinfo: &PatchesInfo,
    p: &Rat3,
    e: Edge,
) -> Result<Option<usize>> {
    let dummy_v = arena.add_or_find_vert(p.clone(), NO_INDEX);
    let dummy = Tri::new([e.v0, e.v1, dummy_v], NO_INDEX, [NO_INDEX; 3]);
    let mut edge_tris = topo
        .edge_tris(e)
        .ok_or_else(|| invariant("sort edge without triangles"))?
        .to_vec();
    edge_tris.push(EXTRA_TRI);
    let view = TriView {
        arena: &*arena,
        tris,
        extra: Some(dummy),
    };
    let t0 = edge_tris[0];
    let sorted = sort_tris_around_edge(&view, e, &edge_tris, t0)?;
    let dummy_index = sorted
        .iter()
        .position(|&t| t == EXTRA_TRI)
        .ok_or_else(|| invariant("dummy triangle lost during edge sort"))?;
    let prev_tri = sorted[(dummy_index + sorted.len() - 1) % sorted.len()];
    if prev_tri == EXTRA_TRI {
        return Err(invariant("dummy triangle is its own neighbour"));
    }
    let (_, prev_flipped) = find_flap_vert(&tris[prev_tri], e)
        .ok_or_else(|| invariant("neighbour triangle lost the sort edge"))?;
    let prev_patch = pinfo.tri_patch[prev_tri]
        .map(|p| &pinfo.patches[p])
        .ok_or_else(|| invariant("triangle without patch"))?;
    Ok(prev_patch.cell(if prev_flipped {
        Side::Below
    } else {
        Side::Above
    }))
}

/// Blender `find_ambient_cell`: the cell outside every other cell. Restricted to the patches of
/// `component` when given. Uses an extreme vertex and a dummy point just beyond it in +x.
pub(super) fn find_ambient_cell(
    arena: &mut Arena,
    tris: &[Tri],
    topo: &Topology,
    pinfo: &PatchesInfo,
    component: Option<&[usize]>,
) -> Result<Option<usize>> {
    let candidates: Vec<usize> = match component {
        None => (0..tris.len()).collect(),
        Some(patches) => patches
            .iter()
            .flat_map(|&p| pinfo.patches[p].tris.iter().copied())
            .collect(),
    };
    let mut extreme: Option<u32> = None;
    for t in candidates {
        for &v in &tris[t].v {
            let better = match extreme {
                None => true,
                Some(best) => arena.vert(v).co[0] > arena.vert(best).co[0],
            };
            if better {
                extreme = Some(v);
            }
        }
    }
    let v_extreme = extreme.ok_or_else(|| invariant("no triangles to find an ambient cell"))?;
    let extreme_co = arena.vert(v_extreme).co.clone();
    // Edge at the extreme vertex with the largest absolute slope in XY. It lies on the convex
    // hull; a vertical edge is taken immediately.
    let mut ehull: Option<Edge> = None;
    let mut max_abs_slope: Option<Rat> = None;
    for &e in topo.vert_edges(v_extreme) {
        let co_other = &arena.vert(e.other(v_extreme)).co;
        let delta_x = &co_other[0] - &extreme_co[0];
        if delta_x.is_zero() {
            ehull = Some(e);
            break;
        }
        let num = &co_other[1] - &extreme_co[1];
        let abs_slope = (num / &delta_x).abs();
        if max_abs_slope.as_ref().is_none_or(|m| abs_slope > *m) {
            ehull = Some(e);
            max_abs_slope = Some(abs_slope);
        }
    }
    let ehull = ehull.ok_or_else(|| invariant("extreme vertex has no edges"))?;
    let mut p_in_ambient = extreme_co;
    p_in_ambient[0] = &p_in_ambient[0] + Rat::one();
    find_cell_for_point_near_edge(arena, tris, topo, pinfo, &p_in_ambient, ehull)
}

/// Blender `find_good_sorting_edge`: an edge at `closestp` on the convex hull of its incident
/// edges, as seen from the line `closestp -> testp`.
fn find_good_sorting_edge(
    arena: &Arena,
    topo: &Topology,
    testp: u32,
    closestp: u32,
) -> Result<Edge> {
    let co_closest = arena.vert(closestp).co.clone();
    let co_test = arena.vert(testp).co.clone();
    let abscissa = sub3(&co_test, &co_closest);
    let axis = (0..3)
        .find(|&a| !abscissa[a].is_zero())
        .ok_or_else(|| invariant("sorting edge query point coincides with the vertex"))?;
    let axis_next = (axis + 1) % 3;
    let axis_next_next = (axis_next + 1) % 3;
    let mut ordinate: Rat3 = [Rat::zero(), Rat::zero(), Rat::zero()];
    ordinate[axis] = abscissa[axis_next].clone();
    ordinate[axis_next] = -abscissa[axis].clone();
    ordinate[axis_next_next] = Rat::zero();
    let normal = cross3(&abscissa, &ordinate);
    let nlen2 = dot3(&normal, &normal);
    let mut max_abs_slope: Option<Rat> = None;
    let mut esort: Option<Edge> = None;
    for &e in topo.vert_edges(closestp) {
        let co_other = &arena.vert(e.other(closestp)).co;
        let evec = sub3(co_other, &co_closest);
        let proj = sub3(&evec, &scale3(&(dot3(&evec, &normal) / &nlen2), &normal));
        let evec_a = dot3(&proj, &abscissa);
        let evec_o = dot3(&proj, &ordinate);
        if evec_a.is_zero() {
            esort = Some(e);
            break;
        }
        let abs_slope = (evec_o / evec_a).abs();
        if max_abs_slope.as_ref().is_none_or(|m| abs_slope > *m) {
            esort = Some(e);
            max_abs_slope = Some(abs_slope);
        }
    }
    esort.ok_or_else(|| invariant("no sorting edge at closest vertex"))
}

/// Corner-pair edge `k -> k + 1` of `tri`, as an [`Edge`].
fn tri_edge(tri: &Tri, k: usize) -> Edge {
    Edge::new(tri.v[k], tri.v[(k + 1) % 3])
}

/// Blender `find_containing_cell`: the cell containing vertex `v`, which is nearest to triangle
/// `t` at `closest`.
fn find_containing_cell(
    arena: &mut Arena,
    tris: &[Tri],
    topo: &Topology,
    pinfo: &PatchesInfo,
    v: u32,
    t: usize,
    closest: Closest,
) -> Result<Option<usize>> {
    let tri = tris[t];
    let etest = match closest {
        Closest::OnVertex(k) => {
            let mut vert_cv = tri.v[k];
            if vert_cv == v {
                vert_cv = tri.v[(k + 1) % 3];
            }
            find_good_sorting_edge(arena, topo, v, vert_cv)?
        }
        Closest::OnEdge(k) => tri_edge(&tri, k),
        Closest::Inside => tri_edge(&tri, 0),
    };
    let co_v = arena.vert(v).co.clone();
    find_cell_for_point_near_edge(arena, tris, topo, pinfo, &co_v, etest)
}

/// Blender `find_component_containers`: components other than `comp` whose cells contain a
/// vertex of `comp`, with the nearest such cell.
fn find_component_containers(
    arena: &mut Arena,
    tris: &[Tri],
    topo: &Topology,
    pinfo: &PatchesInfo,
    components: &[Vec<usize>],
    ambient: &[Option<usize>],
    comp: usize,
    comp_bb: &[BBox],
) -> Result<Vec<ComponentContainer>> {
    let mut ans = Vec::new();
    let test_p = *components[comp]
        .first()
        .ok_or_else(|| invariant("empty patch component"))?;
    let test_t = *pinfo.patches[test_p]
        .tris
        .first()
        .ok_or_else(|| invariant("patch without triangles"))?;
    let test_v = tris[test_t].v[0];
    let test_co = arena.vert(test_v).co.clone();
    for (comp_other, members) in components.iter().enumerate() {
        if comp == comp_other || !comp_bb[comp].might_intersect(&comp_bb[comp_other]) {
            continue;
        }
        let mut nearest: Option<(usize, Rat, Closest)> = None;
        for &p in members {
            for &t in &pinfo.patches[p].tris {
                let tri = tris[t];
                let (d2, closest) = closest_on_tri_to_point(
                    &test_co,
                    &arena.vert(tri.v[0]).co,
                    &arena.vert(tri.v[1]).co,
                    &arena.vert(tri.v[2]).co,
                )?;
                if nearest.as_ref().is_none_or(|(_, best, _)| d2 < *best) {
                    nearest = Some((t, d2, closest));
                }
            }
        }
        let (nearest_t, dist, closest) =
            nearest.ok_or_else(|| invariant("empty patch component during nesting"))?;
        let containing =
            find_containing_cell(arena, tris, topo, pinfo, test_v, nearest_t, closest)?;
        if containing != ambient[comp_other] {
            ans.push(ComponentContainer {
                nearest_cell: containing,
                dist_to_cell: dist,
            });
        }
    }
    Ok(ans)
}

/// Blender `populate_comp_bbs`: per-component boxes padded so that touching boxes overlap.
fn populate_comp_bbs(
    components: &[Vec<usize>],
    tris: &[Tri],
    pinfo: &PatchesInfo,
    arena: &Arena,
) -> Vec<BBox> {
    let mut boxes = vec![BBox::empty(); components.len()];
    let mut max_abs = 0.0_f64;
    for (c, members) in components.iter().enumerate() {
        for &p in members {
            for &t in &pinfo.patches[p].tris {
                for &v in &tris[t].v {
                    let co = arena.vert(v).co_f;
                    boxes[c].include(co);
                    max_abs = max_abs.max(co.abs().max_element());
                }
            }
        }
    }
    let pad = DVec3::splat(1e-9 * (1.0 + max_abs));
    for b in &mut boxes {
        b.min -= pad;
        b.max += pad;
    }
    boxes
}

/// Blender `finish_patch_cell_graph`: connect disconnected components of the patch-cell graph by
/// merging nested components' ambient cells into their containing cells, and outer components'
/// ambient cells into one.
pub(super) fn finish_patch_cell_graph(
    arena: &mut Arena,
    tris: &[Tri],
    topo: &Topology,
    cinfo: &mut CellsInfo,
    pinfo: &mut PatchesInfo,
) -> Result<()> {
    let components = find_patch_components(cinfo, pinfo);
    if components.len() <= 1 {
        return Ok(());
    }
    let mut ambient = Vec::with_capacity(components.len());
    for comp in &components {
        ambient.push(find_ambient_cell(
            arena,
            tris,
            topo,
            pinfo,
            Some(comp.as_slice()),
        )?);
    }
    let comp_bb = populate_comp_bbs(&components, tris, pinfo, arena);
    let mut comp_cont = Vec::with_capacity(components.len());
    for comp in 0..components.len() {
        comp_cont.push(find_component_containers(
            arena,
            tris,
            topo,
            pinfo,
            &components,
            &ambient,
            comp,
            &comp_bb,
        )?);
    }
    let mut outer = Vec::new();
    for (comp, conts) in comp_cont.iter().enumerate() {
        let Some(first) = conts.first() else {
            outer.push(comp);
            continue;
        };
        let mut closest = first;
        for cc in conts.iter().skip(1) {
            if cc.dist_to_cell < closest.dist_to_cell {
                closest = cc;
            }
        }
        let cont_cell = closest
            .nearest_cell
            .ok_or_else(|| invariant("nesting container without a cell"))?;
        let comp_ambient =
            ambient[comp].ok_or_else(|| invariant("component without ambient cell"))?;
        merge_cells(cinfo, pinfo, cont_cell, comp_ambient);
    }
    if outer.len() > 1 {
        let merged =
            ambient[outer[0]].ok_or_else(|| invariant("component without ambient cell"))?;
        for &comp in outer.iter().skip(1) {
            let amb = ambient[comp].ok_or_else(|| invariant("component without ambient cell"))?;
            merge_cells(cinfo, pinfo, merged, amb);
        }
    }
    Ok(())
}
