# Usage: /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --python tests/fixtures/curve_profile_oracle.py
"""Blender 5.2.2 CurveProfile oracle for the bevel custom profile.

Writes curve_profile_oracle.json next to this script. Each case stores the profile points as Blender
holds them after setup (order, locations, handle types and handle positions), the three profile
flags, and the sample array that `BKE_curveprofile_init` builds for each requested segment count.

RNA exposes `CurveProfile.segments` with `totsegments` entries only, while the C array holds
`totsegments + 1` samples, so the samples are read from the struct through ctypes. Layout (checked
against Blender 5.2.2): CurveProfile: short totpoints @0, short segments_len @2,
CurveProfilePoint* points @8, table @0x10, segments @0x18, int flags @0x20. CurveProfilePoint is
40 bytes: float x, y @0, short flag @8, uint8 handle_type_1 @0xa, handle_type_2 @0xb,
float handle_1[2] @0xc, float handle_2[2] @0x14.

Profile type bytes are written after an initial AUTO calculation. Blender's point_calculate_handle
leaves stored handle locations unchanged for FREE and ALIGN sides; authored handles are written
into the CurveProfilePoint struct to verify this behavior independently.
"""
import bpy
import ctypes
import json
import os
import random
import struct


if bpy.app.version != (5, 2, 2):
    raise RuntimeError('need Blender 5.2.2, got ' + str(bpy.app.version))

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'curve_profile_oracle.json')
SEGMENT_COUNTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 32]
AUTO, VECTOR, FREE, ALIGN = 'AUTO', 'VECTOR', 'FREE', 'ALIGNED'
HANDLE_VALUES = {'FREE': 0, 'AUTO': 1, 'VECTOR': 2, 'ALIGNED': 3}
HANDLE_NAMES = {value: name for name, value in HANDLE_VALUES.items()}

def seeded_profile(count, seed):
    """Deterministic profile: x decreasing, y increasing, random handle types on each side."""
    rng = random.Random(seed)
    types = [AUTO, VECTOR, FREE, ALIGN]
    points = []
    for index in range(count):
        t = index / (count - 1)
        x = min(1.0, max(0.0, 1.0 - t + rng.uniform(-0.03, 0.03)))
        y = min(1.0, max(0.0, t + rng.uniform(-0.05, 0.05)))
        points.append(((round(x, 4), round(y, 4)), rng.choice(types), rng.choice(types)))
    return points


ASYMMETRIC_3 = [
    ((0.0, 0.0), AUTO, AUTO),
    ((0.3, 0.85), AUTO, AUTO),
    ((1.0, 1.0), AUTO, AUTO),
]
MIXED_4 = [
    ((1.0, 0.0), AUTO, VECTOR),
    ((0.72, 0.18), VECTOR, AUTO),
    ((0.4, 0.85), AUTO, AUTO),
    ((0.0, 1.0), AUTO, AUTO),
]
NINE_AUTO = [
    ((1.0, 0.0), AUTO, AUTO),
    ((0.9, 0.05), AUTO, AUTO),
    ((0.8, 0.3), AUTO, AUTO),
    ((0.6, 0.5), AUTO, AUTO),
    ((0.5, 0.45), AUTO, AUTO),
    ((0.4, 0.7), AUTO, AUTO),
    ((0.2, 0.8), AUTO, AUTO),
    ((0.1, 0.92), AUTO, AUTO),
    ((0.0, 1.0), AUTO, AUTO),
]
FREE_ALIGN_AUTHORED_5 = [
    ((0.0, 0.0), AUTO, AUTO),
    ((0.25, 0.6), FREE, FREE, (0.16, 0.52), (0.34, 0.71)),
    ((0.55, 0.35), ALIGN, ALIGN, (0.21, 0.22), (0.93, 0.79)),
    ((0.8, 0.9), AUTO, FREE, (0.75, 0.82), (0.91, 0.96)),
    ((1.0, 1.0), VECTOR, VECTOR),
]

# Point specs are (location, handle_type_1, handle_type_2), optionally followed by authored
# handle_1 and handle_2 coordinates.
CASES = [
    ('line_auto_default', [((1.0, 0.0), AUTO, AUTO), ((0.0, 1.0), AUTO, AUTO)], True, False, False),
    ('line_vector', [((1.0, 0.0), VECTOR, VECTOR), ((0.0, 1.0), VECTOR, VECTOR)], True, False, False),
    ('asymmetric_auto_3', ASYMMETRIC_3, True, False, False),
    ('mixed_handles_4', MIXED_4, True, False, False),
    ('free_align_handles_5', FREE_ALIGN_AUTHORED_5, True, False, False),
    ('nine_points_auto', NINE_AUTO, True, False, False),
    ('even_lengths_asymmetric_3', ASYMMETRIC_3, True, False, True),
    ('even_lengths_nine_points', NINE_AUTO, True, False, True),
    ('free_align_even_lengths_5', FREE_ALIGN_AUTHORED_5, True, False, True),
    ('clip_off_out_of_range_3', [
        ((-0.2, 0.1), AUTO, AUTO),
        ((0.5, 0.4), AUTO, AUTO),
        ((1.2, 1.1), AUTO, AUTO),
    ], False, False, False),
    ('free_align_clip_off_5', FREE_ALIGN_AUTHORED_5, False, False, False),
    ('clip_off_even_lengths_asymmetric_3', ASYMMETRIC_3, False, False, True),
    ('free_align_clip_off_even_5', FREE_ALIGN_AUTHORED_5, False, False, True),
    ('clip_off_straight_edges_5', FREE_ALIGN_AUTHORED_5, False, True, False),
    ('clip_off_straight_even_5', FREE_ALIGN_AUTHORED_5, False, True, True),
    ('straight_edges_mixed_4', MIXED_4, True, True, False),
    ('free_align_straight_edges_5', FREE_ALIGN_AUTHORED_5, True, True, False),
    ('straight_edges_even_asymmetric_3', ASYMMETRIC_3, True, True, True),
    ('free_align_straight_even_5', FREE_ALIGN_AUTHORED_5, True, True, True),
    ('straight_edges_nine_points', NINE_AUTO, True, True, False),
    ('vector_chain_fixed_segments_6', [
        ((0.0, 0.0), VECTOR, VECTOR),
        ((0.2, 0.1), VECTOR, VECTOR),
        ((0.5, 0.4), AUTO, AUTO),
        ((0.6, 0.9), VECTOR, VECTOR),
        ((0.8, 0.85), AUTO, AUTO),
        ((1.0, 1.0), VECTOR, VECTOR),
    ], True, False, False),
    ('vector_chain_even_lengths_6', [
        ((0.0, 0.0), VECTOR, VECTOR),
        ((0.2, 0.1), VECTOR, VECTOR),
        ((0.5, 0.4), AUTO, AUTO),
        ((0.6, 0.9), VECTOR, VECTOR),
        ((0.8, 0.85), AUTO, AUTO),
        ((1.0, 1.0), VECTOR, VECTOR),
    ], True, False, True),
    ('long_profile_20_points', seeded_profile(20, 11), True, False, False),
    ('long_profile_20_points_even_lengths', seeded_profile(20, 11), True, False, True),
    ('long_profile_40_points_even_lengths', seeded_profile(40, 23), True, False, True),
    ('seeded_random_types_12', seeded_profile(12, 5), True, False, False),
    ('seeded_random_types_12_straight', seeded_profile(12, 5), True, True, False),
    ('seeded_random_types_25_straight_even', seeded_profile(25, 9), True, True, True),
    ('collinear_vector_chain_10', [
        ((1.0 - index / 9.0, index / 9.0), VECTOR, VECTOR) for index in range(10)
    ], True, False, False),
    ('collinear_vector_chain_10_even_lengths', [
        ((1.0 - index / 9.0, index / 9.0), VECTOR, VECTOR) for index in range(10)
    ], True, False, True),
    ('parabola_vector_41_points', [
        ((1.0 - index / 40.0, (1.0 - index / 40.0) ** 2), VECTOR, VECTOR) for index in range(41)
    ], True, False, False),
    ('parabola_auto_vector_mix_41_straight', [
        ((1.0 - index / 40.0, (1.0 - index / 40.0) ** 2), AUTO if index % 3 else VECTOR,
         VECTOR if index % 4 else AUTO) for index in range(41)
    ], True, True, False),
]


def profile_address(profile):
    return profile.as_pointer()


def points_address(profile):
    return struct.unpack('<Q', ctypes.string_at(profile_address(profile) + 8, 8))[0]


def read_profile(profile):
    """Reads the CurveProfile struct as Blender holds it."""
    base = profile_address(profile)
    total_points = struct.unpack('<h', ctypes.string_at(base, 2))[0]
    segments_len = struct.unpack('<h', ctypes.string_at(base + 2, 2))[0]
    points_ptr = struct.unpack('<Q', ctypes.string_at(base + 8, 8))[0]
    segments_ptr = struct.unpack('<Q', ctypes.string_at(base + 0x18, 8))[0]
    flags = struct.unpack('<i', ctypes.string_at(base + 0x20, 4))[0]
    points = []
    for index in range(total_points):
        raw = ctypes.string_at(points_ptr + 40 * index, 28)
        x, y, flag, h1_type, h2_type, h1x, h1y, h2x, h2y = struct.unpack('<ffhBBffff', raw)
        points.append({
            'location': [x, y],
            'handle_type_1': HANDLE_NAMES[h1_type],
            'handle_type_2': HANDLE_NAMES[h2_type],
            'handle_1': [h1x, h1y],
            'handle_2': [h2x, h2y],
        })
    samples = None
    if segments_ptr and segments_len:
        raw = ctypes.string_at(segments_ptr, 40 * (segments_len + 1))
        samples = [list(struct.unpack('<ff', raw[40 * k:40 * k + 8])) for k in range(segments_len + 1)]
    return {
        'totpoints': total_points,
        'segments_len': segments_len,
        'flags': flags,
        'points': points,
        'segments': samples,
    }


def point_spec_parts(point_spec):
    if len(point_spec) == 3:
        return point_spec[0], point_spec[1], point_spec[2], None
    if len(point_spec) == 5:
        return point_spec[0], point_spec[1], point_spec[2], (point_spec[3], point_spec[4])
    raise RuntimeError('point spec must have location and handle types, optionally handle locations')


def write_handle_types(profile, points):
    """Writes both handle-type bytes of every point (offsets 0xa and 0xb of each point)."""
    base = points_address(profile)
    for index, point_spec in enumerate(points):
        _, handle_1, handle_2, _ = point_spec_parts(point_spec)
        ctypes.memmove(
            base + 40 * index + 0xa,
            bytes([HANDLE_VALUES[handle_1], HANDLE_VALUES[handle_2]]),
            2,
        )


def write_handle_locations(profile, points):
    """Writes explicitly authored handle coordinates before Blender's per-type recalculation."""
    base = points_address(profile)
    for index, point_spec in enumerate(points):
        _, _, _, locations = point_spec_parts(point_spec)
        if locations is not None:
            h1, h2 = locations
            ctypes.memmove(
                base + 40 * index + 0xc,
                struct.pack('<ffff', *h1, *h2),
                16,
            )


def build_profile(points, use_clip, straight, even):
    obj = bpy.data.objects.new('curve_profile_oracle', bpy.data.meshes.new('curve_profile_oracle'))
    profile = obj.modifiers.new('bevel', 'BEVEL').custom_profile
    profile.use_clip = use_clip
    profile.use_sample_straight_edges = straight
    profile.use_sample_even_lengths = even
    while len(profile.points) < len(points):
        profile.points.add(0.5, 0.5)
    for point, point_spec in zip(profile.points, points):
        location, _, _, _ = point_spec_parts(point_spec)
        point.location = location
    # Compute default AUTO handles from final locations before authoring handle types/locations.
    profile.initialize(2)
    write_handle_types(profile, points)
    write_handle_locations(profile, points)
    return profile


def main():
    out_cases = []
    for name, points, use_clip, straight, even in CASES:
        profile = build_profile(points, use_clip, straight, even)
        stored = read_profile(profile)
        samples = {}
        for count in SEGMENT_COUNTS:
            profile.initialize(count)
            state = read_profile(profile)
            if state['segments_len'] != count:
                raise RuntimeError('segments_len mismatch for %s at %d' % (name, count))
            expected_types = [(point_spec[1], point_spec[2]) for point_spec in points]
            if [(p['handle_type_1'], p['handle_type_2']) for p in state['points']] != expected_types:
                raise RuntimeError('handle types changed during initialize for %s' % name)
            samples[str(count)] = state['segments']
            stored = state
        out_cases.append({
            'name': name,
            'use_clip': use_clip,
            'use_sample_straight_edges': straight,
            'use_sample_even_lengths': even,
            'points': stored['points'],
            'samples': samples,
        })
    document = {
        'generator': 'tests/fixtures/curve_profile_oracle.py',
        'blender_version': list(bpy.app.version),
        'segment_counts': SEGMENT_COUNTS,
        'cases': out_cases,
    }
    with open(FIXTURE, 'w') as handle:
        json.dump(document, handle, indent=1)
        handle.write('\n')
    print('wrote', FIXTURE, len(out_cases), 'cases')


main()
