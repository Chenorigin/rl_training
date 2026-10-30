"""Simulator-independent stair geometry and vectorised safe support regions."""
import numpy as np
import torch
import trimesh

from .settings import WidthSettings


def stair_mesh(g, s: WidthSettings):
    """Closed boxes for boards (finite thickness), open void between treads.

    This represents board geometry and an effective contact response, not the
    vibrational/acoustic response of a particular manufactured hollow staircase.
    Coordinates: origin at middle of start platform, course along +X.
    """
    start = s.start_length / 2
    spans = [(-start, start, 0.)]
    spans += [(start + i*g.depth, start + (i+1)*g.depth, (i+1)*g.height)
              for i in range(g.steps)]
    spans += [(start + g.steps*g.depth, s.tile_length-start, g.steps*g.height)]
    meshes = []
    for x0, x1, z in spans:
        thickness = z+s.floor_depth if g.closed_riser else s.board_thickness
        box = trimesh.creation.box((x1-x0, g.width, thickness))
        box.apply_translation(((x0+x1)/2, 0., z-thickness/2))
        meshes.append(box)
    floor = trimesh.creation.box((s.tile_length, s.tile_width, .05))
    floor.apply_translation((s.tile_length/2-start, 0., -s.floor_depth-.025))
    meshes.append(floor)
    return trimesh.util.concatenate(meshes)


def wheel_margins(s, reference):
    """Explicit FL, FR, HL, HR ordering; independent front/rear settings."""
    return (reference.new_tensor([s.front_tread_margin]*2+[s.rear_tread_margin]*2),
            reference.new_tensor([s.front_lateral_margin]*2+[s.rear_lateral_margin]*2))


def foothold_regions(position, axis, height, depth, width, steps, s: WidthSettings):
    """position/axis [N,4,3], course geometry [N].

    Side clearance uses the entire capsule envelope. Along X use the bottom
    support patch + projected capsule axis + explicit margin. The physical
    wheel radius is already used in support-height checking by the caller.
    Final tread is continuous with the upper landing (no fictitious seam).
    """
    x, y = position[..., 0], position[..., 1]
    start = s.start_length/2
    index = torch.floor((x-start)/depth[:, None]).long().clamp_min(0)
    index = torch.minimum(index, steps[:, None]-1)
    lo = start + index*depth[:, None]
    hi = lo+depth[:, None]
    lo = torch.where(x < start, torch.full_like(lo, -start), lo)
    hi = torch.where(x < start, torch.full_like(hi, start), hi)
    hi = torch.where((index == steps[:, None]-1) & (x >= start),
                     torch.full_like(hi, s.tile_length-start), hi)
    top = torch.where(x < start, torch.zeros_like(x), (index+1)*height[:, None])
    # A loaded wheel perched on the next tread's lip can have its centre
    # behind that tread. Assign the support plane by height as well as XY,
    # otherwise it appears "unsupported" and escapes the foothold penalty.
    bottom = position[..., 2]-s.wheel_radius-s.wheel_half_length*axis[..., 2].abs()
    level = torch.round(bottom/height[:, None]).long().clamp_min(0)
    level = torch.minimum(level, steps[:, None])
    candidate_top = level*height[:, None]
    candidate_lo = torch.where(level == 0, torch.full_like(x, -start), start+(level-1)*depth[:, None])
    candidate_hi = torch.where(level == 0, torch.full_like(x, start), start+level*depth[:, None])
    candidate_hi = torch.where(level == steps[:, None],
                               torch.full_like(x, s.tile_length-start), candidate_hi)
    reach = s.wheel_radius+s.wheel_half_length*axis[..., 0].abs()
    plausible = ((bottom-candidate_top).abs() < .05) & (x >= candidate_lo-reach) & (x <= candidate_hi+reach)
    lo = torch.where(plausible, candidate_lo, lo)
    hi = torch.where(plausible, candidate_hi, hi)
    top = torch.where(plausible, candidate_top, top)
    extent_x = s.contact_patch_radius + s.wheel_half_length*axis[..., 0].abs()
    extent_y = s.wheel_radius + s.wheel_half_length*axis[..., 1].abs()
    tread_margin, lateral_margin = wheel_margins(s, position)
    forward_margin = torch.minimum(x-lo, hi-x)-extent_x-tread_margin
    side_margin = width[:, None]/2-y.abs()-extent_y-lateral_margin
    return forward_margin, side_margin, top


def capsule_contact_velocity(velocity, omega, axis, radius=.09, half_length=.027):
    """Bottom support velocity for horizontal treads (not a riser normal).

    Handles wheel rotation; ideal rolling has zero slip, COM translation alone
    is not slip. Ground is static in this task.
    """
    offset = torch.zeros_like(velocity)
    offset[..., 2] = -radius
    offset -= half_length * torch.sign(axis[..., 2:3])*axis
    return velocity + torch.linalg.cross(omega, offset, dim=-1)


def face_profile_ids(mesh, num_cols, tile_width, profile_count):
    """TerrainGenerator centred columns, each geometry contains all profiles."""
    column = np.floor((mesh.triangles_center[:, 1] + num_cols*tile_width/2) / tile_width)
    return np.clip(column.astype(np.int64), 0, num_cols-1) % profile_count
