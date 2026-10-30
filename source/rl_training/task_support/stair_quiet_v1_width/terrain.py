"""Prebuilt physical width rows, preserving separate contact collision meshes."""
from dataclasses import replace
import numpy as np
import torch
from isaaclab.terrains import TerrainImporter
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass
from .settings import GEOMETRIES, PROFILES, cases
from .geometry import stair_mesh, face_profile_ids
from .settings import WidthSettings, width_for_difficulty


def width_stairs(difficulty, cfg):
    s = cfg.settings
    geometry = replace(GEOMETRIES[cfg.geometry_index], width=width_for_difficulty(difficulty, s))
    mesh = stair_mesh(geometry, s)
    origin = np.array((s.start_length/2, s.tile_width/2, 0.))
    mesh.apply_translation(origin)
    return [mesh], origin


@configclass
class WidthStairsCfg(SubTerrainBaseCfg):
    function = width_stairs
    geometry_index: int = 0
    profile_index: int = 0
    settings: WidthSettings = WidthSettings()


class WidthTerrainImporter(TerrainImporter):
    def import_mesh(self, name, mesh):
        import isaaclab.sim as sim
        from isaaclab.terrains.utils import create_prim_from_mesh
        from pxr import UsdPhysics, PhysxSchema, UsdShade
        import omni.usd
        super().import_mesh(name, mesh)
        stage = omni.usd.get_context().get_stage()
        root = f"{self.cfg.prim_path}/{name}"
        UsdPhysics.CollisionAPI(stage.GetPrimAtPath(root+"/mesh")).GetCollisionEnabledAttr().Set(False)
        generator = self.cfg.terrain_generator
        if generator.num_cols != len(cases()):
            raise ValueError("Width course requires one column per original geometry/contact case")
        selected = face_profile_ids(mesh, generator.num_cols, generator.size[1], len(PROFILES))
        self.contact_receipt = []
        for i, p in enumerate(PROFILES):
            contact_root = f"{self.cfg.prim_path}/contact_{i}"
            path = contact_root+"/physicsMaterial"
            part = mesh.submesh([np.flatnonzero(selected == i)], append=True, repair=False)
            material = sim.RigidBodyMaterialCfg(static_friction=p.static_friction,
                dynamic_friction=p.dynamic_friction, restitution=p.restitution,
                friction_combine_mode="multiply", restitution_combine_mode="multiply",
                compliant_contact_stiffness=p.stiffness, compliant_contact_damping=p.damping)
            create_prim_from_mesh(contact_root, part, physics_material=material)
            prim = stage.GetPrimAtPath(path)
            physics, api = UsdPhysics.MaterialAPI(prim), PhysxSchema.PhysxMaterialAPI(prim)
            actual = [physics.GetStaticFrictionAttr().Get(), physics.GetDynamicFrictionAttr().Get(),
                physics.GetRestitutionAttr().Get(), api.GetCompliantContactStiffnessAttr().Get(),
                api.GetCompliantContactDampingAttr().Get()]
            bound, _ = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(contact_root+"/mesh")).ComputeBoundMaterial(materialPurpose="physics")
            if not np.allclose(actual, [p.static_friction, p.dynamic_friction, p.restitution, p.stiffness, p.damping], rtol=1e-6) or str(bound.GetPath()) != path:
                raise RuntimeError("Width terrain material/binding readback mismatch")
            self.contact_receipt.append({"name": p.name, "values": actual, "triangles": len(part.faces),
                                         "material_path": path, "binding_verified": True})


def neutral_wheel_material(env):
    """Make ground profile parameters effective, rather than hidden products."""
    robot = env.scene["robot"]
    view = robot.root_physx_view
    ids, names = robot.find_bodies([f"{leg}_wheel" for leg in ("fl", "fr", "hl", "hr")], preserve_order=True)
    counts = [robot._physics_sim_view.create_rigid_body_view(p).max_shapes for p in view.link_paths[0]]
    if sum(counts) != view.max_shapes or len(ids) != 4:
        raise ValueError("Wheel material shape mapping changed")
    shapes = [j for i in ids for j in range(sum(counts[:i]), sum(counts[:i+1]))]
    material = view.get_material_properties()
    before = material.clone()
    material[:, shapes, :] = torch.tensor([1., 1., 0.])
    env_ids = torch.arange(env.num_envs, dtype=torch.int32, device="cpu")
    view.set_material_properties(material, env_ids)
    actual = view.get_material_properties()
    if not torch.allclose(actual[:, shapes], material[:, shapes], atol=1e-6, rtol=0):
        raise RuntimeError("Neutral wheel material runtime readback failed")
    other = sorted(set(range(view.max_shapes))-set(shapes))
    if other and not torch.equal(actual[:, other], before[:, other]):
        raise RuntimeError("Non-wheel material changed")
    env.tail_wheel_material_receipt = {"shape_ids": shapes, "values": [1., 1., 0.], "runtime_readback": True}
