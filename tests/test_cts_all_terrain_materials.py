"""CPU USD authoring checks; the cooked PhysX ray gate runs in validation."""
import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
import trimesh

Usd = pytest.importorskip('pxr.Usd', reason='Requires Isaac USD libraries on PYTHONPATH')
from pxr import Gf, PhysxSchema, Plug, UsdGeom, UsdPhysics, UsdShade

# Standalone USD has no Kit extension manager to register PhysX's schemas.
for resource in (Path(PhysxSchema.__file__).parents[2]/'plugins').glob('*/resources'):
    Plug.Registry().RegisterPlugins(str(resource))


@pytest.fixture
def importer(monkeypatch):
    stage = Usd.Stage.CreateInMemory()
    sim = ModuleType('isaaclab.sim')

    class MaterialCfg:
        def __init__(self, **values):
            self.__dict__.update(values)

        @staticmethod
        def func(path, cfg):
            material = UsdShade.Material.Define(stage, path)
            physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
            physics.CreateStaticFrictionAttr().Set(cfg.static_friction)
            physics.CreateDynamicFrictionAttr().Set(cfg.dynamic_friction)
            physics.CreateRestitutionAttr().Set(cfg.restitution)
            extra = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
            extra.CreateCompliantContactStiffnessAttr().Set(cfg.compliant_contact_stiffness)
            extra.CreateCompliantContactDampingAttr().Set(cfg.compliant_contact_damping)

    sim.RigidBodyMaterialCfg = MaterialCfg
    sim.SimulationContext = SimpleNamespace(instance=lambda: SimpleNamespace(
        cfg=SimpleNamespace(enable_scene_query_support=True)))
    isaac = ModuleType('isaaclab'); isaac.sim = sim
    isaac.terrains = ModuleType('isaaclab.terrains')
    isaac.terrains.utils = ModuleType('isaaclab.terrains.utils')
    def create_mesh(path, source, translation, physics_material):
        mesh = UsdGeom.Mesh.Define(stage, path+'/mesh')
        mesh.CreatePointsAttr().Set(source.vertices)
        mesh.CreateFaceVertexIndicesAttr().Set(source.faces.flatten())
        mesh.CreateFaceVertexCountsAttr().Set(np.full(len(source.faces), 3))
        mesh.AddTranslateOp().Set(Gf.Vec3d(*translation))
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr().Set(True)
        physics_material.func(path+'/physicsMaterial', physics_material)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
            UsdShade.Material.Get(stage, path+'/physicsMaterial'),
            bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose='physics')
    isaac.terrains.utils.create_prim_from_mesh = create_mesh
    omni = ModuleType('omni'); omni.usd = ModuleType('omni.usd')
    omni.usd.get_context = lambda: SimpleNamespace(get_stage=lambda: stage)
    for name, module in [('isaaclab', isaac), ('isaaclab.sim', sim), ('isaaclab.terrains', isaac.terrains),
                         ('isaaclab.terrains.utils', isaac.terrains.utils), ('omni', omni), ('omni.usd', omni.usd)]:
        monkeypatch.setitem(sys.modules, name, module)

    class TerrainImporter:
        def import_mesh(self, name, source):
            root = f'{self.cfg.prim_path}/{name}'
            mesh = UsdGeom.Mesh.Define(stage, root+'/mesh')
            mesh.CreatePointsAttr().Set(source.vertices)
            mesh.CreateFaceVertexIndicesAttr().Set(source.faces.flatten())
            mesh.CreateFaceVertexCountsAttr().Set(np.full(len(source.faces), 3))
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr().Set(True)
            material = UsdShade.Material.Define(stage, root+'/physicsMaterial')
            UsdPhysics.MaterialAPI.Apply(material.GetPrim())
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose='physics')

    path = Path(__file__).parents[1]/'source/rl_training/task_support/cts/all_terrain_isaac.py'
    node = next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.ClassDef) and node.name == 'AllTerrainImporter')
    node.body = [part for part in node.body if isinstance(part, ast.FunctionDef)
                 and part.name in ('import_ground_plane', 'validate_contacts')]
    scope = dict(TerrainImporter=TerrainImporter, np=np, torch=torch,
                 __name__=__name__, __package__='task_support.cts')
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    value = scope['AllTerrainImporter']()
    value.cfg = SimpleNamespace(prim_path='/World/ground')
    return value, stage


def tiny_atlas(count):
    meshes = [trimesh.Trimesh(vertices=[[3*i, 0, 0], [3*i+1, 0, 0], [3*i, 1, 0]],
        faces=[[0, 1, 2]], process=False) for i in range(count)]
    records = [dict(stiffness=4e4+i*123.45, damping=500.+i*.537) for i in range(count)]
    origins = np.array([[3*i, 0, 0] for i in range(count)])
    return SimpleNamespace(mesh=trimesh.util.concatenate(meshes), tile_meshes=meshes, records=records, origins=origins)


def test_1120_continuous_single_material_colliders_match_the_ray_atlas(importer):
    value, stage = importer
    value.atlas = tiny_atlas(1120)
    value.import_ground_plane('terrain')
    colliders = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    enabled = [p for p in colliders if UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()]
    assert len(enabled) == 1120
    assert len(colliders) == 1121
    assert len(value.contact_receipt) == 1120
    for prim in enabled:
        tile = int(prim.GetParent().GetName().split('_')[1])
        mesh = UsdGeom.Mesh(prim)
        assert UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == 'none'
        assert not UsdGeom.Subset.GetGeomSubsets(mesh)
        points = np.asarray(mesh.GetPointsAttr().Get())
        assert np.max(np.abs(points)) <= 1.
        world = np.array([mesh.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).Transform(Gf.Vec3d(*map(float,p))) for p in points])
        np.testing.assert_allclose(world, value.atlas.tile_meshes[tile].vertices, atol=1e-6)
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose='physics')
        assert str(material.GetPath()) == f'/World/ground/contact_{tile}/physicsMaterial'
        actual = PhysxSchema.PhysxMaterialAPI(material.GetPrim()).GetCompliantContactStiffnessAttr().Get()
        assert actual == pytest.approx(value.atlas.records[tile]['stiffness'], rel=1e-6)
    assert not UsdPhysics.CollisionAPI(stage.GetPrimAtPath('/World/ground/terrain/mesh')).GetCollisionEnabledAttr().Get()


def test_mismatched_concatenation_order_is_rejected(importer):
    value, _ = importer
    value.atlas = tiny_atlas(3)
    value.atlas.mesh.faces[[0, 1]] = value.atlas.mesh.faces[[1, 0]]
    with pytest.raises(RuntimeError, match='face order'):
        value.import_ground_plane('terrain')


def test_missing_material_record_cannot_leave_uncovered_faces(importer):
    value, _ = importer
    value.atlas = tiny_atlas(3)
    value.atlas.records.pop()
    with pytest.raises(RuntimeError, match='requires its own continuous material'):
        value.import_ground_plane('terrain')


@pytest.mark.parametrize('fault', [None, 'material', 'height', 'face', 'collision', 'miss'])
def test_cooked_contact_gate_rejects_each_mismatch(importer, monkeypatch, fault):
    """Exercise gate decisions; this stub does not replace the real PhysX test."""
    from task_support.cts.all_terrain_geometry import TorchSurface, build_atlas
    value, _ = importer
    value.atlas = build_atlas(dict(samples_per_level=1, geometry_levels=1, grid_resolution=.25))
    value.import_ground_plane('terrain')
    xy = np.stack((value.atlas.bounds[:, 0]+.3, value.atlas.bounds[:, 2]+.3), -1)
    heights, _ = TorchSurface(value.atlas, 'cpu').sample(torch.arange(len(xy)), xy)
    calls = []

    def raycast(start, direction, distance):
        index = len(calls)
        calls.append(start)
        assert direction == (0., 0., -1.) and distance > 2.
        receipt = value.contact_receipt[index]
        result = dict(hit=True, collision=receipt['collision_path'], material=receipt['material_path'],
            position=(start[0], start[1], heights[index].item()), faceIndex=receipt['face_start'])
        if index == 2:
            if fault == 'material': result['material'] = '/wrong/material'
            if fault == 'height': result['position'] = (start[0], start[1], heights[index].item()+.01)
            if fault == 'face': result['faceIndex'] = receipt['face_end']
            if fault == 'collision': result['collision'] = '/World/robot'
            if fault == 'miss': result['hit'] = False
        return result

    physx = ModuleType('omni.physx')
    physx.get_physx_scene_query_interface = lambda: SimpleNamespace(raycast_closest=raycast)
    monkeypatch.setitem(sys.modules, 'omni.physx', physx)
    if fault is None:
        result = value.validate_contacts()
        assert result['tiles_checked'] == 10 and result['cooked_materials_verified']
        assert result['height_error_max_m'] == 0.
    else:
        with pytest.raises(RuntimeError, match='Physical terrain'):
            value.validate_contacts()
        assert len(calls) == 3


def test_headless_configuration_enables_actual_physx_scene_queries():
    path = Path(__file__).parents[1]/'source/rl_training/task_support/cts/all_terrain_isaac.py'
    function = next(node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'configure_all_terrain')
    assignments = [node for node in function.body if isinstance(node, ast.Assign)
                   and any(ast.unparse(target) == 'cfg.sim.enable_scene_query_support' for target in node.targets)]
    assert len(assignments) == 1
    cfg = SimpleNamespace(sim=SimpleNamespace(enable_scene_query_support=False))
    exec(compile(ast.Module(body=assignments, type_ignores=[]), str(path), 'exec'), dict(cfg=cfg))
    assert cfg.sim.enable_scene_query_support is True


def test_explicit_collision_groups_avoid_env_id_expansion_without_robot_crosstalk():
    path = Path(__file__).parents[1]/'source/rl_training/task_support/cts/all_terrain_isaac.py'
    function = next(node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'configure_all_terrain')
    targets = {'cfg.scene.replicate_physics', 'cfg.scene.filter_collisions'}
    assignments = [node for node in function.body if isinstance(node, ast.Assign)
                   and any(ast.unparse(target) in targets for target in node.targets)]
    assert len(assignments) == 2
    cfg = SimpleNamespace(scene=SimpleNamespace(replicate_physics=True, filter_collisions=False))
    exec(compile(ast.Module(body=assignments, type_ignores=[]), str(path), 'exec'), dict(cfg=cfg))
    assert cfg.scene.replicate_physics is False
    assert cfg.scene.filter_collisions is True


def test_disabled_query_instrument_is_reported_before_false_collider_miss(importer, monkeypatch):
    value, _ = importer
    sys.modules['isaaclab.sim'].SimulationContext = SimpleNamespace(instance=lambda: SimpleNamespace(
        cfg=SimpleNamespace(enable_scene_query_support=False)))
    physx = ModuleType('omni.physx')
    def unexpected_query():
        pytest.fail('The disabled query instrument must never be read')
    physx.get_physx_scene_query_interface = unexpected_query
    monkeypatch.setitem(sys.modules, 'omni.physx', physx)
    with pytest.raises(RuntimeError, match='requires sim.enable_scene_query_support=True'):
        value.validate_contacts()
