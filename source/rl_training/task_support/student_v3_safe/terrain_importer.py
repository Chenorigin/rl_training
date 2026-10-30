"""Prebuilt physical width rows, preserving separate contact collision meshes."""
from dataclasses import replace
import numpy as np
from isaaclab.terrains import TerrainImporter
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass
from task_support.stair_quiet_v1_width.settings import PROFILES
from .terrain_geometry import geometries, cases, course_mesh
from task_support.stair_quiet_v1_width.geometry import face_profile_ids
from task_support.stair_quiet_v1_width.settings import WidthSettings, width_for_difficulty


def student_stairs(difficulty, cfg):
    s = cfg.settings
    geometry = replace(geometries(cfg.course)[cfg.geometry_index], width=width_for_difficulty(difficulty, s))
    mesh = course_mesh(geometry, s)
    origin = np.array((s.start_length/2, s.tile_width/2, 0.))
    mesh.apply_translation(origin)
    return [mesh], origin


@configclass
class StudentStairsCfg(SubTerrainBaseCfg):
    function = student_stairs
    course: dict = None
    geometry_index: int = 0
    profile_index: int = 0
    settings: WidthSettings = WidthSettings()


class StudentTerrainImporter(TerrainImporter):
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
        if generator.num_cols != len(cases(next(iter(generator.sub_terrains.values())).course)):
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


def configure_course(cfg,course):
    from copy import deepcopy
    from .terrain_geometry import validate_course
    from .terrain_course import reset_course
    from .terrain_observations import teacher_safety
    validate_course(course)
    cfg.student_terrain_course=deepcopy(course)
    cfg.stair_tail=replace(cfg.stair_tail,tile_length=course['tile_length'])
    cfg.stair_tail.validate()
    cfg.episode_length_s=course['episode_length_s']
    s=cfg.stair_tail
    terrain=cfg.scene.terrain
    terrain.class_type=StudentTerrainImporter
    generator=terrain.terrain_generator
    table=geometries(course)
    from .terrain_geometry import surfaces
    longest=max(surfaces(g,s)[1] for g in table)
    pauses=sum(cfg.student_idle_practice[k] for k in ('initial_hold_s','move_s','stop_s'))
    minimum_time=(longest+s.body_half_length)/s.speed_range[0]+pauses+s.success_hold_s
    if cfg.episode_length_s<=minimum_time:
        raise ValueError('Episode duration cannot cover slowest full course plus stop practice')
    generator.num_rows,generator.num_cols=len(s.widths_m),len(table)*len(PROFILES)
    generator.size=(s.tile_length,s.tile_width)
    generator.sub_terrains={f'{g.name}__{p.name}':StudentStairsCfg(
        proportion=1.,geometry_index=gi,profile_index=pi,settings=s,course=deepcopy(course))
        for gi,g in enumerate(table) for pi,p in enumerate(PROFILES)}
    cfg.events.randomize_reset_base.func=reset_course
    cfg.observations.safety.safety.func=teacher_safety
