"""Measured normals for the student capture grid; teacher rays stay unchanged.

Use Isaac's mesh query at each existing hit, not a finite-difference estimate
across stair edges. The short second cast avoids copying the upstream sensor's
pose/drift implementation. This is simulator acquisition metadata, never an
extra policy input or a real-robot geometry oracle.
"""
import torch
from isaaclab.sensors import RayCaster
from isaaclab.utils.warp import raycast_mesh


class SurfaceRayCaster(RayCaster):
    def _update_buffers_impl(self, env_ids):
        super()._update_buffers_impl(env_ids)
        hits = self._data.ray_hits_w[env_ids]
        physical = hits.clone()
        physical[..., 2] -= self.ray_cast_drift[env_ids, 2].unsqueeze(-1)
        finite = torch.isfinite(physical).all(-1)
        starts = torch.where(finite[..., None], physical, 0.).clone()
        starts[..., 2] += .01
        directions = torch.zeros_like(starts)
        directions[..., 2] = -1.
        measured, _, normals, _ = raycast_mesh(
            starts, directions, mesh=self.meshes[self.cfg.mesh_prim_paths[0]],
            max_dist=.02, return_normal=True)
        matched = finite & torch.isfinite(measured).all(-1)
        matched &= (measured-physical).abs().amax(-1) < 1.e-4
        if not hasattr(self._data, 'ray_normals_w'):
            self._data.ray_normals_w = torch.full_like(self._data.ray_hits_w, torch.nan)
        self._data.ray_normals_w[env_ids] = torch.where(matched[..., None], normals, torch.nan)
