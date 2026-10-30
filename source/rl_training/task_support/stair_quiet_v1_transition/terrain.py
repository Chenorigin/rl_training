"""Independent 35-case mesh/material importer; old 30-case source stays pinned."""
from dataclasses import replace
import numpy as np
from isaaclab.terrains import TerrainImporter
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass
from task_support.stair_quiet_v1_width.geometry import stair_mesh, face_profile_ids
from task_support.stair_quiet_v1_width.settings import width_for_difficulty
from .settings import GEOMETRIES, PROFILES, cases, TransitionSettings


def transition_stairs(difficulty, cfg):
    s = cfg.settings
    geometry = replace(GEOMETRIES[cfg.geometry_index], width=width_for_difficulty(difficulty, s))
    mesh = stair_mesh(geometry, s)
    origin = np.array((s.start_length/2, s.tile_width/2, 0.))
    mesh.apply_translation(origin)
    return [mesh], origin


@configclass
class TransitionStairsCfg(SubTerrainBaseCfg):
    function = transition_stairs
    geometry_index: int = 0
    profile_index: int = 0
    settings: TransitionSettings = TransitionSettings()


class TransitionTerrainImporter(TerrainImporter):
    def import_mesh(self, name, mesh):
        import isaaclab.sim as sim
        from isaaclab.terrains.utils import create_prim_from_mesh
        from pxr import UsdPhysics, PhysxSchema, UsdShade
        import omni.usd
        super().import_mesh(name, mesh)
        generator = self.cfg.terrain_generator
        if generator.num_cols != len(cases()):
            raise ValueError("Transition terrain requires all 35 geometry/material columns")
        stage = omni.usd.get_context().get_stage()
        root = f"{self.cfg.prim_path}/{name}"
        UsdPhysics.CollisionAPI(stage.GetPrimAtPath(root+"/mesh")).GetCollisionEnabledAttr().Set(False)
        selected = face_profile_ids(mesh, generator.num_cols, generator.size[1], len(PROFILES))
        self.contact_receipt = []
        for i, p in enumerate(PROFILES):
            contact_root = f"{self.cfg.prim_path}/contact_{i}"
            material_path = contact_root+"/physicsMaterial"
            part = mesh.submesh([np.flatnonzero(selected == i)], append=True, repair=False)
            material = sim.RigidBodyMaterialCfg(static_friction=p.static_friction,
                dynamic_friction=p.dynamic_friction, restitution=p.restitution,
                friction_combine_mode="multiply", restitution_combine_mode="multiply",
                compliant_contact_stiffness=p.stiffness, compliant_contact_damping=p.damping)
            create_prim_from_mesh(contact_root, part, physics_material=material)
            prim = stage.GetPrimAtPath(material_path)
            physical, api = UsdPhysics.MaterialAPI(prim), PhysxSchema.PhysxMaterialAPI(prim)
            actual = [physical.GetStaticFrictionAttr().Get(), physical.GetDynamicFrictionAttr().Get(),
                physical.GetRestitutionAttr().Get(), api.GetCompliantContactStiffnessAttr().Get(),
                api.GetCompliantContactDampingAttr().Get()]
            bound, _ = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(contact_root+"/mesh")).ComputeBoundMaterial(materialPurpose="physics")
            expected = [p.static_friction, p.dynamic_friction, p.restitution, p.stiffness, p.damping]
            if not np.allclose(actual, expected, rtol=1e-6) or str(bound.GetPath()) != material_path:
                raise RuntimeError("Transition physical material readback/binding mismatch")
            self.contact_receipt.append(dict(name=p.name, values=actual, triangles=len(part.faces),
                material_path=material_path, binding_verified=True))
