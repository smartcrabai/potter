use crate::model::{
    Extrapolation, FCurve, FCurveAutoSmoothing, FCurveHandleType, Interpolation, Keyframe,
};

const AUTO_HANDLE_FACTOR: f32 = 2.5614;
const HANDLE_TIME_THRESHOLD: f32 = 0.001;

#[derive(Clone, Copy)]
struct Point {
    co: [f32; 2],
    left: [f32; 2],
    right: [f32; 2],
    left_type: FCurveHandleType,
    right_type: FCurveHandleType,
    auto_locked: bool,
}

pub(super) fn prepare_key_insert(curve: &mut FCurve, insert_at: usize, key: &mut Keyframe) {
    recalculate(curve);
    key.handle_left = Some([key.frame - 1.0, key.value]);
    key.handle_right = Some([key.frame + 1.0, key.value]);

    if insert_at == 0 || insert_at >= curve.keyframes.len() {
        return;
    }
    let previous_index = insert_at - 1;
    let next_index = insert_at;
    if curve.keyframes[previous_index].interpolation != Interpolation::Bezier
        || key.interpolation != Interpolation::Bezier
    {
        return;
    }

    let inserted = point(key);
    let inserted_auto = is_autoh(inserted);
    let inserted_vector = inserted.left_type == FCurveHandleType::Vector
        && inserted.right_type == FCurveHandleType::Vector;
    let previous_point = point(&curve.keyframes[previous_index]);
    let next_point = point(&curve.keyframes[next_index]);
    let previous_auto =
        is_autoh(previous_point) || previous_point.right_type == FCurveHandleType::Vector;
    let next_auto = is_autoh(next_point) || next_point.left_type == FCurveHandleType::Vector;
    if (inserted_auto || inserted_vector) && previous_auto && next_auto {
        return;
    }

    let start = previous_point;
    let end = next_point;
    let Some(start_right) = curve.keyframes[previous_index].handle_right else {
        return;
    };
    let Some(end_left) = curve.keyframes[next_index].handle_left else {
        return;
    };
    let mut start_right = start_right.map(|value| value as f32);
    let mut end_left = end_left.map(|value| value as f32);
    let end_x = end.co[0] - start.co[0];
    let first_x = (start.co[0] - start_right[0]).abs();
    let second_x = (end.co[0] - end_left[0]).abs();
    if !is_zero(first_x + second_x) {
        if first_x > end_x {
            let factor = end_x / first_x;
            start_right = [
                start.co[0] - factor * (start.co[0] - start_right[0]),
                start.co[1] - factor * (start.co[1] - start_right[1]),
            ];
        }
        if second_x > end_x {
            let factor = end_x / second_x;
            end_left = [
                end.co[0] - factor * (end.co[0] - end_left[0]),
                end.co[1] - factor * (end.co[1] - end_left[1]),
            ];
        }
    }
    curve.keyframes[previous_index].handle_right = Some(start_right.map(f64::from));
    curve.keyframes[next_index].handle_left = Some(end_left.map(f64::from));

    let target_x = key.frame as f32;
    let mut lower = 0.0_f32;
    let mut upper = 1.0_f32;
    for _ in 0..40 {
        let parameter = f32::midpoint(lower, upper);
        if cubic(
            start.co[0],
            start_right[0],
            end_left[0],
            end.co[0],
            parameter,
        ) < target_x
        {
            lower = parameter;
        } else {
            upper = parameter;
        }
    }
    let t = f32::midpoint(lower, upper);
    let split_a = [
        lerp(start.co[0], start_right[0], t),
        lerp(start.co[1], start_right[1], t),
    ];
    let split_b = [
        lerp(start_right[0], end_left[0], t),
        lerp(start_right[1], end_left[1], t),
    ];
    let split_c = [
        lerp(end_left[0], end.co[0], t),
        lerp(end_left[1], end.co[1], t),
    ];
    let split_d = [
        lerp(split_a[0], split_b[0], t),
        lerp(split_a[1], split_b[1], t),
    ];
    let split_e = [
        lerp(split_b[0], split_c[0], t),
        lerp(split_b[1], split_c[1], t),
    ];
    let split_point = [
        lerp(split_d[0], split_e[0], t),
        lerp(split_d[1], split_e[1], t),
    ];
    let difference = [target_x - split_point[0], key.value as f32 - split_point[1]];

    curve.keyframes[previous_index].handle_right = Some(split_a.map(f64::from));
    curve.keyframes[next_index].handle_left = Some(split_c.map(f64::from));
    key.handle_left = Some([
        f64::from(split_d[0] + difference[0]),
        f64::from(split_d[1] + difference[1]),
    ]);
    key.handle_right = Some([
        f64::from(split_e[0] + difference[0]),
        f64::from(split_e[1] + difference[1]),
    ]);

    if inserted_auto {
        let auto_neighbour = previous_auto || next_auto;
        let smoothing_keeps_auto = auto_neighbour
            && curve.auto_smoothing.unwrap_or_default() == FCurveAutoSmoothing::ContAccel
            && ((key.frame as f32 - key.handle_left.unwrap_or_default()[0] as f32)
                - (key.frame as f32 - curve.keyframes[previous_index].frame as f32) / 3.0)
                .abs()
                < HANDLE_TIME_THRESHOLD;
        if !smoothing_keeps_auto {
            key.handle_left_type = Some(FCurveHandleType::Aligned);
            key.handle_right_type = Some(FCurveHandleType::Aligned);
        }
    }
}

pub(super) fn recalculate(curve: &mut FCurve) {
    if curve.keyframes.len() < 2 {
        return;
    }

    for key in &mut curve.keyframes {
        if key.handle_left_type.is_none() && key.handle_left.is_none() {
            key.handle_left_type = Some(FCurveHandleType::Auto);
        }
        if key.handle_right_type.is_none() && key.handle_right.is_none() {
            key.handle_right_type = Some(FCurveHandleType::Auto);
        }
    }
    if curve.keyframes.iter().any(|key| {
        key.handle_left_type.is_some_and(|handle_type| {
            matches!(
                handle_type,
                FCurveHandleType::Aligned | FCurveHandleType::Free
            )
        }) && key.handle_left.is_none()
            || key.handle_right_type.is_some_and(|handle_type| {
                matches!(
                    handle_type,
                    FCurveHandleType::Aligned | FCurveHandleType::Free
                )
            }) && key.handle_right.is_none()
    }) {
        let mut initial = curve.clone();
        for key in &mut initial.keyframes {
            key.handle_left_type = Some(FCurveHandleType::Auto);
            key.handle_right_type = Some(FCurveHandleType::Auto);
        }
        recalculate_core(&mut initial);
        for (key, seed) in curve.keyframes.iter_mut().zip(initial.keyframes) {
            if key.handle_left.is_none() {
                key.handle_left = seed.handle_left;
            }
            if key.handle_right.is_none() {
                key.handle_right = seed.handle_right;
            }
        }
    }
    recalculate_core(curve);
}

fn recalculate_core(curve: &mut FCurve) {
    let count = curve.keyframes.len();
    if count < 2 {
        return;
    }

    let cycle = is_cycle(curve)
        && is_autoh(point(&curve.keyframes[0]))
        && is_autoh(point(&curve.keyframes[count - 1]));
    let first = point(&curve.keyframes[0]);
    let last = point(&curve.keyframes[count - 1]);
    let cycle_delta = [last.co[0] - first.co[0], last.co[1] - first.co[1]];
    let mut points = curve.keyframes.iter().map(point).collect::<Vec<_>>();

    for index in 0..count {
        let current = points[index];
        points[index].left[0] =
            current.left[0].min(next_down(current.co[0] - HANDLE_TIME_THRESHOLD));
        points[index].right[0] =
            current.right[0].max(next_up(current.co[0] + HANDLE_TIME_THRESHOLD));

        let previous = if index > 0 {
            Some(points[index - 1].co)
        } else if cycle {
            let mut previous = points[count - 2].co;
            previous[0] -= cycle_delta[0];
            previous[1] -= cycle_delta[1];
            Some(previous)
        } else {
            None
        };
        let next = if index + 1 < count {
            Some(points[index + 1].co)
        } else if cycle {
            let mut next = points[1].co;
            next[0] += cycle_delta[0];
            next[1] += cycle_delta[1];
            Some(next)
        } else {
            None
        };
        calculate_key_handles(
            &mut points[index],
            previous,
            next,
            curve.auto_smoothing.unwrap_or_default(),
        );

        if is_autoh(points[index])
            && !cycle
            && (index == 0 || index + 1 == count)
            && curve.extrapolation == Extrapolation::Constant
        {
            points[index].left[1] = points[index].co[1];
            points[index].right[1] = points[index].co[1];
            points[index].auto_locked = true;
        }

        if index > 0 && points[index - 1].co[0] >= points[index].co[0] {
            points[index - 1].auto_locked = true;
            points[index].auto_locked = true;
        }
    }

    if cycle && (points[0].auto_locked || points[count - 1].auto_locked) {
        points[0].left[1] = points[0].co[1];
        points[0].right[1] = points[0].co[1];
        points[count - 1].left[1] = points[count - 1].co[1];
        points[count - 1].right[1] = points[count - 1].co[1];
        points[0].auto_locked = true;
        points[count - 1].auto_locked = true;
    }

    if curve.auto_smoothing.unwrap_or_default() == FCurveAutoSmoothing::ContAccel {
        smooth_handles(&mut points, cycle);
    }

    for (key, point) in curve.keyframes.iter_mut().zip(points) {
        key.handle_left = Some(point.left.map(f64::from));
        key.handle_right = Some(point.right.map(f64::from));
    }
}

fn calculate_key_handles(
    point: &mut Point,
    previous: Option<[f32; 2]>,
    next: Option<[f32; 2]>,
    smoothing: FCurveAutoSmoothing,
) {
    point.auto_locked = false;
    let clamp_extremes = previous.is_some() && next.is_some();
    let previous_co = previous.unwrap_or_else(|| {
        let next = next.unwrap_or(point.co);
        [2.0 * point.co[0] - next[0], 2.0 * point.co[1] - next[1]]
    });
    let next_co = next.unwrap_or_else(|| {
        [
            2.0 * point.co[0] - previous_co[0],
            2.0 * point.co[1] - previous_co[1],
        ]
    });

    if is_auto(point.left_type) || is_auto(point.right_type) {
        let before = [point.co[0] - previous_co[0], point.co[1] - previous_co[1]];
        let after = [next_co[0] - point.co[0], next_co[1] - point.co[1]];
        let mut length_before = before[0];
        let mut length_after = after[0];
        if is_zero(length_before) {
            length_before = 1.0;
        }
        if is_zero(length_after) {
            length_after = 1.0;
        }
        let tangent = [
            after[0] / length_after + before[0] / length_before,
            after[1] / length_after + before[1] / length_before,
        ];
        let mut length = if smoothing == FCurveAutoSmoothing::None {
            tangent[0]
        } else {
            6.0 / AUTO_HANDLE_FACTOR
        };
        length *= AUTO_HANDLE_FACTOR;
        if !is_zero(length) {
            let mut left_violate = false;
            let mut right_violate = false;
            if smoothing == FCurveAutoSmoothing::None {
                length_before = length_before.min(5.0 * length_after);
                length_after = length_after.min(5.0 * length_before);
            }
            if is_auto(point.left_type) {
                let handle_length = length_before / length;
                point.left = [
                    point.co[0] - tangent[0] * handle_length,
                    point.co[1] - tangent[1] * handle_length,
                ];
                if point.left_type == FCurveHandleType::AutoClamped && clamp_extremes {
                    let previous_delta = previous_co[1] - point.co[1];
                    let next_delta = next_co[1] - point.co[1];
                    if (previous_delta <= 0.0 && next_delta <= 0.0)
                        || (previous_delta >= 0.0 && next_delta >= 0.0)
                    {
                        point.left[1] = point.co[1];
                        point.auto_locked = true;
                    } else if (previous_delta <= 0.0 && previous_co[1] > point.left[1])
                        || (previous_delta > 0.0 && previous_co[1] < point.left[1])
                    {
                        point.left[1] = previous_co[1];
                        left_violate = true;
                    }
                }
            }
            if is_auto(point.right_type) {
                let handle_length = length_after / length;
                point.right = [
                    point.co[0] + tangent[0] * handle_length,
                    point.co[1] + tangent[1] * handle_length,
                ];
                if point.right_type == FCurveHandleType::AutoClamped && clamp_extremes {
                    let previous_delta = previous_co[1] - point.co[1];
                    let next_delta = next_co[1] - point.co[1];
                    if (previous_delta <= 0.0 && next_delta <= 0.0)
                        || (previous_delta >= 0.0 && next_delta >= 0.0)
                    {
                        point.right[1] = point.co[1];
                        point.auto_locked = true;
                    } else if (next_delta <= 0.0 && next_co[1] > point.right[1])
                        || (next_delta > 0.0 && next_co[1] < point.right[1])
                    {
                        point.right[1] = next_co[1];
                        right_violate = true;
                    }
                }
            }
            if left_violate || right_violate {
                let left_x = point.left[0] - point.co[0];
                let right_x = point.co[0] - point.right[0];
                if left_violate {
                    point.right[1] = point.co[1] + (point.co[1] - point.left[1]) / left_x * right_x;
                } else {
                    point.left[1] = point.co[1] + (point.co[1] - point.right[1]) / right_x * left_x;
                }
            }
        }
    }

    if point.left_type == FCurveHandleType::Vector {
        point.left = [
            point.co[0] - (point.co[0] - previous_co[0]) / 3.0,
            point.co[1] - (point.co[1] - previous_co[1]) / 3.0,
        ];
    }
    if point.right_type == FCurveHandleType::Vector {
        point.right = [
            point.co[0] + (next_co[0] - point.co[0]) / 3.0,
            point.co[1] + (next_co[1] - point.co[1]) / 3.0,
        ];
    }

    if point.left_type == FCurveHandleType::Free
        || point.right_type == FCurveHandleType::Free
        || (point.left_type != FCurveHandleType::Aligned
            && point.right_type != FCurveHandleType::Aligned)
    {
        return;
    }
    let mut length_before = distance(point.co, point.left);
    let mut length_after = distance(point.co, point.right);
    if is_zero(length_before) {
        length_before = 1.0;
    }
    if is_zero(length_after) {
        length_after = 1.0;
    }
    let ratio = length_before / length_after;
    if point.right_type == FCurveHandleType::Aligned {
        point.right = [
            point.co[0] + (1.0 / ratio) * (point.co[0] - point.left[0]),
            point.co[1] + (1.0 / ratio) * (point.co[1] - point.left[1]),
        ];
    }
    if point.left_type == FCurveHandleType::Aligned {
        point.left = [
            point.co[0] + ratio * (point.co[0] - point.right[0]),
            point.co[1] + ratio * (point.co[1] - point.right[1]),
        ];
    }
}

fn smooth_handles(points: &mut [Point], cycle: bool) {
    let total = points.len();
    let cycle = cycle && free_auto(points[0]) && free_auto(points[total - 1]);
    if cycle {
        let break_at = (1..total - 1).find(|&index| !free_auto(points[index]));
        let Some(search_base) = break_at else {
            smooth_run(points, 0, total, true);
            return;
        };
        let mut start = search_base;
        let mut count = 1;
        let mut current = search_base + 1;
        for _ in 1..total {
            if current == total - 1 {
                current = 0;
            }
            if free_auto(points[current]) {
                count += 1;
            } else {
                smooth_run(points, start, count + 1, true);
                start = current;
                count = 1;
            }
            current += 1;
        }
        if count > 1 {
            smooth_run(points, start, count, true);
        }
        return;
    }

    let mut start = 0;
    let mut count = 1;
    for index in 1..total {
        if free_auto(points[index]) {
            count += 1;
        } else {
            smooth_run(points, start, count + 1, false);
            start = index;
            count = 1;
        }
    }
    if count > 1 {
        smooth_run(points, start, count, false);
    }
}
fn smooth_run(points: &mut [Point], start: usize, count: usize, cycle: bool) {
    if count < 2 {
        return;
    }
    let total = points.len();
    let full_cycle = start == 0 && count == total && cycle;
    let last_index = if start + count > total {
        start + count - total
    } else {
        start + count - 1
    };
    let solve_first = solve_end_handle(points[start], points[start].right_type, start == 0);
    let solve_last = solve_end_handle(
        points[last_index],
        points[last_index].left_type,
        start + count == total,
    );
    if count == 2 && !full_cycle && solve_first == solve_last {
        return;
    }

    let mut dx = vec![0.0; count];
    let mut dy = vec![0.0; count];
    let mut ratio = vec![1.0; count];
    let mut a = vec![0.0; count];
    let mut b = vec![0.0; count];
    let mut c = vec![0.0; count];
    let mut d = vec![0.0; count];
    let mut h = vec![0.0; count];
    let mut maximum = vec![f32::MAX; count];
    let mut minimum = vec![-f32::MAX; count];

    let mut current = start + 1;
    for index in 1..count {
        let previous = current - 1;
        dx[index] = points[current].co[0] - points[previous].co[0];
        dy[index] = points[current].co[1] - points[previous].co[1];
        if cycle && current == total - 1 {
            current = 0;
        }
        current += 1;
    }
    if full_cycle {
        dx[0] = dx[count - 1];
        dy[0] = dy[count - 1];
        ratio[0] = dx[1] / dx[0];
        ratio[count - 1] = ratio[0];
    } else {
        ratio[0] = 1.0;
        ratio[count - 1] = 1.0;
    }
    for index in 1..count - 1 {
        ratio[index] = dx[index + 1] / dx[index];
    }

    let mut clamped_current = has_auto_clamped(points[start]);
    let mut current = start + 1;
    for index in 1..count {
        let clamped_previous = clamped_current;
        clamped_current = has_auto_clamped(points[current]);
        if cycle && current == total - 1 {
            current = 0;
            clamped_current |= has_auto_clamped(points[current]);
        }
        clamp_bounds(
            &mut maximum,
            &mut minimum,
            index - 1,
            dy[index],
            clamped_previous,
        );
        clamp_bounds(
            &mut maximum,
            &mut minimum,
            index,
            dy[index] * ratio[index],
            clamped_current,
        );
        current += 1;
    }

    let mut solve_count = count;
    let mut first_adjust = 0.0;
    let mut last_adjust = 0.0;
    if full_cycle {
        solve_count = count - 1;
        minimum[0] = minimum[0].max(minimum[count - 1]);
        maximum[0] = maximum[0].min(maximum[count - 1]);
        equation_continuous(&mut a, &mut b, &mut c, &mut d, &dy, &ratio, 0);
    } else {
        if solve_first {
            equation_no_accel_right(&mut a, &mut b, &mut c, &mut d, &dy, &ratio, 0);
        } else {
            let mut handle = [
                points[start].right[0] - points[start].co[0],
                points[start].right[1] - points[start].co[1],
            ];
            first_adjust = handle_adjust(&mut handle, dx[1]);
            lock_unknown(&mut a, &mut b, &mut c, &mut d, 0, handle[1]);
        }
        if solve_last {
            equation_no_accel_left(&mut a, &mut b, &mut c, &mut d, &dy, &ratio, count - 1);
        } else {
            let mut handle = [
                points[last_index].co[0] - points[last_index].left[0],
                points[last_index].co[1] - points[last_index].left[1],
            ];
            last_adjust = handle_adjust(&mut handle, dx[count - 1]);
            lock_unknown(&mut a, &mut b, &mut c, &mut d, count - 1, handle[1]);
        }
    }
    for index in 1..count - 1 {
        equation_continuous(&mut a, &mut b, &mut c, &mut d, &dy, &ratio, index);
    }
    if !full_cycle {
        if count > 2 || solve_last {
            b[1] += ratio[1] * first_adjust;
        }
        if count > 2 || solve_first {
            b[count - 2] += last_adjust;
        }
    }

    if solve_with_limits(
        &mut a,
        &mut b,
        &mut c,
        &mut d,
        &mut h,
        &minimum,
        &maximum,
        solve_count,
    ) {
        if full_cycle {
            h[count - 1] = h[0];
        }
        let mut current = start + 1;
        for index in 1..count - 1 {
            let end = current == total - 1;
            output_handle(points, current, false, -h[index] / ratio[index], end);
            if end {
                current = 0;
            }
            output_handle(points, current, true, h[index], end);
            current += 1;
        }
        if solve_first {
            output_handle(points, start, true, h[0], start == 0);
        }
        if solve_last {
            output_handle(
                points,
                last_index,
                false,
                -h[count - 1] / ratio[count - 1],
                start + count == total,
            );
        }
    }
}

fn solve_with_limits(
    a: &mut [f32],
    b: &mut [f32],
    c: &mut [f32],
    d: &mut [f32],
    h: &mut [f32],
    minimum: &[f32],
    maximum: &[f32],
    solve_count: usize,
) -> bool {
    let original_a = a.to_vec();
    let original_b = b.to_vec();
    let original_c = c.to_vec();
    let original_d = d.to_vec();
    let mut locked = vec![false; solve_count];
    let mut unlocks = vec![0_u8; solve_count];

    loop {
        if !solve_cyclic_tridiagonal(a, b, c, d, h, solve_count) {
            return false;
        }
        let mut overshoot = false;
        let mut unlocked = false;
        let mut all = false;
        let mut locked_any = false;
        loop {
            for index in 0..solve_count {
                if h[index] >= minimum[index] && h[index] <= maximum[index] {
                    continue;
                }
                overshoot = true;
                let target = if h[index] > maximum[index] {
                    maximum[index]
                } else {
                    minimum[index]
                };
                if !is_zero(target) || all {
                    locked[index] = true;
                    lock_unknown(a, b, c, d, index, target);
                    locked_any = true;
                }
            }
            all = true;
            if !overshoot || locked_any {
                break;
            }
        }
        if !locked_any {
            for index in 0..solve_count {
                if !locked[index] || unlocks[index] >= 2 {
                    continue;
                }
                let relax = relax_direction(
                    &original_a,
                    &original_b,
                    &original_c,
                    &original_d,
                    &h[..solve_count],
                    index,
                );
                if (relax > 0.0 && h[index] < maximum[index])
                    || (relax < 0.0 && h[index] > minimum[index])
                {
                    a[index] = original_a[index];
                    b[index] = original_b[index];
                    c[index] = original_c[index];
                    d[index] = original_d[index];
                    locked[index] = false;
                    unlocks[index] += 1;
                    unlocked = true;
                }
            }
        }
        if !overshoot && !unlocked {
            return h.iter().take(solve_count).all(|value| value.is_finite());
        }
    }
}

fn solve_cyclic_tridiagonal(
    a: &[f32],
    b: &[f32],
    c: &[f32],
    d: &[f32],
    x: &mut [f32],
    count: usize,
) -> bool {
    if count == 0 {
        return false;
    }
    if count == 1 {
        x[0] = d[0] / (a[0] + b[0] + c[0]);
        return x[0].is_finite();
    }
    if count == 2 {
        let lower = [0.0, a[1] + c[1]];
        let upper = [a[0] + c[0], 0.0];
        return solve_tridiagonal(&lower, b, &upper, d, x, count);
    }
    let first = a[0];
    let last = c[count - 1];
    if is_zero(first) && is_zero(last) {
        return solve_tridiagonal(a, b, c, d, x, count);
    }
    let mut adjusted_b = b.to_vec();
    adjusted_b[0] -= first;
    adjusted_b[count - 1] -= last;
    let mut right_hand_side = vec![0.0; count];
    right_hand_side[0] = first;
    right_hand_side[count - 1] = last;
    let mut temporary = vec![0.0; count];
    let success = solve_tridiagonal(a, &adjusted_b, c, &right_hand_side, &mut temporary, count)
        && solve_tridiagonal(a, &adjusted_b, c, d, x, count);
    if !success {
        return false;
    }
    let coefficient = (x[0] + x[count - 1]) / (1.0 + temporary[0] + temporary[count - 1]);
    for index in 0..count {
        x[index] -= coefficient * temporary[index];
    }
    x.iter().take(count).all(|value| value.is_finite())
}

fn solve_tridiagonal(
    a: &[f32],
    b: &[f32],
    c: &[f32],
    d: &[f32],
    x: &mut [f32],
    count: usize,
) -> bool {
    if count == 0 {
        return false;
    }
    let mut upper = vec![0.0_f64; count];
    let mut rhs = vec![0.0_f64; count];
    upper[0] = f64::from(c[0]) / f64::from(b[0]);
    rhs[0] = f64::from(d[0]) / f64::from(b[0]);
    for index in 1..count {
        let denominator = f64::from(b[index]) - f64::from(a[index]) * upper[index - 1];
        upper[index] = f64::from(c[index]) / denominator;
        rhs[index] = (f64::from(d[index]) - f64::from(a[index]) * rhs[index - 1]) / denominator;
    }
    let mut current = rhs[count - 1];
    x[count - 1] = current as f32;
    for index in (0..count - 1).rev() {
        current = rhs[index] - upper[index] * current;
        x[index] = current as f32;
    }
    x.iter().take(count).all(|value| value.is_finite())
}

fn output_handle(points: &mut [Point], index: usize, right: bool, offset: f32, endpoint: bool) {
    let point = &mut points[index];
    let handle_type = if right {
        point.right_type
    } else {
        point.left_type
    };
    if !is_auto(handle_type) && handle_type != FCurveHandleType::Vector {
        return;
    }

    let mut new_value = if right { point.right } else { point.left };
    new_value[1] = point.co[1] + offset;
    if right {
        point.right = new_value;
    } else {
        point.left = new_value;
    }
    let other_type = if right {
        point.left_type
    } else {
        point.right_type
    };
    if other_type == FCurveHandleType::Aligned {
        let other = if right { point.left } else { point.right };
        let own_length = distance(point.co, new_value);
        let other_length = distance(point.co, other);
        let direction = [point.co[0] - new_value[0], point.co[1] - new_value[1]];
        let aligned = [
            point.co[0] + direction[0] * (other_length / own_length),
            point.co[1] + direction[1] * (other_length / own_length),
        ];
        if right {
            point.left = aligned;
        } else {
            point.right = aligned;
        }
    } else if endpoint && (is_auto(other_type) || other_type == FCurveHandleType::Vector) {
        let mirrored = [
            2.0 * point.co[0] - new_value[0],
            2.0 * point.co[1] - new_value[1],
        ];
        if right {
            point.left = mirrored;
        } else {
            point.right = mirrored;
        }
    }
}

fn clamp_bounds(
    maximum: &mut [f32],
    minimum: &mut [f32],
    index: usize,
    delta: f32,
    no_overshoot: bool,
) {
    if delta > 0.0 {
        if no_overshoot {
            maximum[index] = maximum[index].min(delta);
        }
        if no_overshoot {
            minimum[index] = 0.0;
        }
    } else if delta < 0.0 {
        if no_overshoot {
            maximum[index] = 0.0;
        }
        if no_overshoot {
            minimum[index] = minimum[index].max(delta);
        }
    } else if no_overshoot {
        maximum[index] = 0.0;
        minimum[index] = 0.0;
    }
}

fn equation_continuous(
    a: &mut [f32],
    b: &mut [f32],
    c: &mut [f32],
    d: &mut [f32],
    dy: &[f32],
    ratio: &[f32],
    index: usize,
) {
    a[index] = ratio[index] * ratio[index];
    b[index] = 2.0 * (ratio[index] + 1.0);
    c[index] = 1.0 / ratio[index + 1];
    d[index] = dy[index] * ratio[index] * ratio[index] + dy[index + 1];
}

fn equation_no_accel_right(
    a: &mut [f32],
    b: &mut [f32],
    c: &mut [f32],
    d: &mut [f32],
    dy: &[f32],
    ratio: &[f32],
    index: usize,
) {
    a[index] = 0.0;
    b[index] = 2.0;
    c[index] = 1.0 / ratio[index + 1];
    d[index] = dy[index + 1];
}

fn equation_no_accel_left(
    a: &mut [f32],
    b: &mut [f32],
    c: &mut [f32],
    d: &mut [f32],
    dy: &[f32],
    ratio: &[f32],
    index: usize,
) {
    a[index] = ratio[index] * ratio[index];
    b[index] = 2.0 * ratio[index];
    c[index] = 0.0;
    d[index] = dy[index] * ratio[index] * ratio[index];
}

fn lock_unknown(
    a: &mut [f32],
    b: &mut [f32],
    c: &mut [f32],
    d: &mut [f32],
    index: usize,
    value: f32,
) {
    a[index] = 0.0;
    b[index] = 1.0;
    c[index] = 0.0;
    d[index] = value;
}

fn handle_adjust(handle: &mut [f32; 2], delta_x: f32) -> f32 {
    let factor = delta_x / (handle[0] + delta_x / 3.0);
    if factor < 1.0 {
        handle[0] *= factor;
        handle[1] *= factor;
    }
    1.0 - 3.0 * handle[0] / delta_x
}

fn relax_direction(a: &[f32], b: &[f32], c: &[f32], d: &[f32], h: &[f32], index: usize) -> f32 {
    let count = h.len();
    let state = a[index] * h[(index + count - 1) % count]
        + b[index] * h[index]
        + c[index] * h[(index + 1) % count]
        - d[index];
    -state * b[index]
}

fn point(key: &Keyframe) -> Point {
    let co = [key.frame as f32, key.value as f32];
    let left = key
        .handle_left
        .map_or(co, |handle| handle.map(|value| value as f32));
    let right = key
        .handle_right
        .map_or(co, |handle| handle.map(|value| value as f32));
    Point {
        co,
        left,
        right,
        left_type: key
            .handle_left_type
            .unwrap_or(if key.handle_left.is_some() {
                FCurveHandleType::Free
            } else {
                FCurveHandleType::Auto
            }),
        right_type: key
            .handle_right_type
            .unwrap_or(if key.handle_right.is_some() {
                FCurveHandleType::Free
            } else {
                FCurveHandleType::Auto
            }),
        auto_locked: false,
    }
}

fn is_autoh(point: Point) -> bool {
    is_auto(point.left_type) && is_auto(point.right_type)
}

fn has_auto_clamped(point: Point) -> bool {
    point.left_type == FCurveHandleType::AutoClamped
        || point.right_type == FCurveHandleType::AutoClamped
}

fn is_auto(handle_type: FCurveHandleType) -> bool {
    matches!(
        handle_type,
        FCurveHandleType::Auto | FCurveHandleType::AutoClamped
    )
}

fn free_auto(point: Point) -> bool {
    is_autoh(point) && !point.auto_locked
}

fn solve_end_handle(point: Point, handle_type: FCurveHandleType, endpoint: bool) -> bool {
    handle_type == FCurveHandleType::Vector
        || (endpoint && is_auto(handle_type) && !point.auto_locked)
}

fn is_cycle(curve: &FCurve) -> bool {
    let Some(modifier) = curve.modifiers.first() else {
        return false;
    };
    if modifier.modifier_type != "CYCLES"
        || modifier
            .properties
            .get("mute")
            .and_then(serde_json::Value::as_bool)
            == Some(true)
        || modifier
            .properties
            .get("use_restricted_range")
            .and_then(serde_json::Value::as_bool)
            == Some(true)
        || modifier
            .properties
            .get("use_influence")
            .and_then(serde_json::Value::as_bool)
            == Some(true)
        || modifier
            .properties
            .get("cycles_before")
            .and_then(serde_json::Value::as_i64)
            .unwrap_or_default()
            != 0
        || modifier
            .properties
            .get("cycles_after")
            .and_then(serde_json::Value::as_i64)
            .unwrap_or_default()
            != 0
    {
        return false;
    }
    ["mode_before", "mode_after"].iter().all(|field| {
        matches!(
            modifier
                .properties
                .get(*field)
                .and_then(serde_json::Value::as_str),
            None | Some("REPEAT" | "REPEAT_OFFSET")
        )
    })
}

fn distance(first: [f32; 2], second: [f32; 2]) -> f32 {
    ((first[0] - second[0]).powi(2) + (first[1] - second[1]).powi(2)).sqrt()
}

fn cubic(start: f32, control_a: f32, control_b: f32, end: f32, t: f32) -> f32 {
    let inverse = 1.0 - t;
    inverse.powi(3) * start
        + 3.0 * inverse.powi(2) * t * control_a
        + 3.0 * inverse * t.powi(2) * control_b
        + t.powi(3) * end
}

fn lerp(start: f32, end: f32, factor: f32) -> f32 {
    start + (end - start) * factor
}

fn is_zero(value: f32) -> bool {
    value.to_bits().trailing_zeros() >= 31
}

fn next_up(value: f32) -> f32 {
    if value.is_nan() || value.to_bits() == f32::INFINITY.to_bits() {
        value
    } else if is_zero(value) {
        f32::from_bits(1)
    } else if value > 0.0 {
        f32::from_bits(value.to_bits() + 1)
    } else {
        f32::from_bits(value.to_bits() - 1)
    }
}

fn next_down(value: f32) -> f32 {
    if value.is_nan() || value.to_bits() == f32::NEG_INFINITY.to_bits() {
        value
    } else if is_zero(value) {
        f32::from_bits(0x8000_0001)
    } else if value > 0.0 {
        f32::from_bits(value.to_bits() - 1)
    } else {
        f32::from_bits(value.to_bits() + 1)
    }
}
