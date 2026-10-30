"""CTS all-terrain simulator wiring. Import only after Isaac app startup."""
from copy import deepcopy
import numpy as np
import torch
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.envs.mdp import UniformVelocityCommand
from isaaclab.managers import EventTermCfg, RewardTermCfg, TerminationTermCfg
from isaaclab.utils import configclass
from .all_terrain_course import course
from . import all_terrain_terms as terms


@configclass
class AllTerrainImporterCfg(TerrainImporterCfg):
    settings: dict = {}
    cohort: dict = {}


class AllTerrainImporter(TerrainImporter):
    def __init__(self,cfg):
        from .all_terrain_geometry import build_atlas
        self.atlas = build_atlas(cfg.settings['geometry'],cfg.settings.get('wall_practice'))
        super().__init__(cfg)
        from .practice import cohort_mask
        from .all_terrain_curriculum import allocate_courses
        teacher = cohort_mask(cfg.num_envs,cfg.cohort['teacher_fraction'],cfg.cohort['cohort_seed'],self.device)
        courses = allocate_courses(teacher,cfg.settings['fractions'],cfg.cohort['cohort_seed'])
        tiles = []
        for i,c in enumerate(courses.tolist()):
            pool = np.flatnonzero((self.atlas.course_ids == c) & (self.atlas.geometry_levels == 0))
            tiles.append(int(pool[i%len(pool)]))
        self.initial_tiles = torch.tensor(tiles,device=self.device,dtype=torch.long)
        origins = torch.tensor(self.atlas.origins,device=self.device,dtype=torch.float32)
        self.env_origins = origins[self.initial_tiles].clone()
        self.terrain_origins = origins[None]
        self.terrain_levels = torch.zeros(cfg.num_envs,device=self.device,dtype=torch.long)
        self.terrain_types = self.initial_tiles.clone()
        self.max_terrain_level = cfg.settings['geometry']['geometry_levels']

    def import_ground_plane(self,name,size=None):
        import isaaclab.sim as sim
        from isaaclab.terrains.utils import create_prim_from_mesh
        from pxr import UsdPhysics,PhysxSchema,UsdShade
        import omni.usd
        # The ray atlas and per-tile physical shapes share identical triangles.
        # GPU contact generation does not support triangle material subsets.
        super().import_mesh(name,self.atlas.mesh)
        stage = omni.usd.get_context().get_stage()
        root = f'{self.cfg.prim_path}/{name}'
        ray_collision = UsdPhysics.CollisionAPI(stage.GetPrimAtPath(root+'/mesh'))
        ray_collision.GetCollisionEnabledAttr().Set(False)
        if ray_collision.GetCollisionEnabledAttr().Get() is not False:
            raise RuntimeError('Merged ray atlas must not create duplicate physical contacts')
        if len(self.atlas.tile_meshes) != len(self.atlas.records):
            raise RuntimeError('Each physical terrain tile requires its own continuous material')
        self.contact_receipt = []
        next_face,next_vertex = 0,0
        for index,(mesh,record) in enumerate(zip(self.atlas.tile_meshes,self.atlas.records)):
            path = f'{self.cfg.prim_path}/contact_{index}'
            offset = np.r_[self.atlas.origins[index,:2],0.]
            local_mesh = mesh.copy()
            local_mesh.apply_translation(-offset)
            material = sim.RigidBodyMaterialCfg(static_friction=1.,dynamic_friction=1.,restitution=0.,
                friction_combine_mode='multiply',restitution_combine_mode='multiply',
                compliant_contact_stiffness=record['stiffness'],compliant_contact_damping=record['damping'])
            end_face = next_face+len(mesh.faces)
            if not np.array_equal(self.atlas.mesh.faces[next_face:end_face],mesh.faces+next_vertex):
                raise RuntimeError('Physical tile face order differs from the ray atlas')
            create_prim_from_mesh(path,local_mesh,translation=offset,physics_material=material)
            collision_prim = stage.GetPrimAtPath(path+'/mesh')
            UsdPhysics.MeshCollisionAPI.Apply(collision_prim).CreateApproximationAttr().Set('none')
            material_path = path+'/physicsMaterial'
            prim = stage.GetPrimAtPath(material_path)
            physics,api = UsdPhysics.MaterialAPI(prim),PhysxSchema.PhysxMaterialAPI(prim)
            actual = [physics.GetStaticFrictionAttr().Get(),physics.GetDynamicFrictionAttr().Get(),
                api.GetCompliantContactStiffnessAttr().Get(),api.GetCompliantContactDampingAttr().Get()]
            bound,_ = UsdShade.MaterialBindingAPI(collision_prim).ComputeBoundMaterial(materialPurpose='physics')
            if not np.allclose(actual,[1.,1.,record['stiffness'],record['damping']],rtol=1e-6) or str(bound.GetPath()) != material_path:
                raise RuntimeError('Continuous physical terrain material/binding mismatch')
            self.contact_receipt.append(dict(tile=index,values=actual,binding_verified=True,
                material_path=material_path,collision_path=str(collision_prim.GetPath()),
                face_start=0,face_end=len(mesh.faces)))
            next_face,next_vertex = end_face,next_vertex+len(mesh.vertices)
        if next_face != len(self.atlas.mesh.faces):
            raise RuntimeError('Physical terrain tiles do not cover every ray atlas face')

    def validate_contacts(self):
        """Check cooked colliders after simulation startup, with scene queries enabled."""
        from isaaclab.sim import SimulationContext
        from omni.physx import get_physx_scene_query_interface
        from .all_terrain_geometry import TorchSurface
        simulation = SimulationContext.instance()
        if simulation is None or not simulation.cfg.enable_scene_query_support:
            raise RuntimeError('Physical terrain validation requires sim.enable_scene_query_support=True before startup')
        query = get_physx_scene_query_interface()
        # Corners lie outside the central route and typical reset footprints.
        xy = np.stack((self.atlas.bounds[:,0]+.3,self.atlas.bounds[:,2]+.3),-1)
        expected,_ = TorchSurface(self.atlas,'cpu').sample(torch.arange(len(xy)),torch.as_tensor(xy))
        max_error = 0.
        for index,(point,height,receipt) in enumerate(zip(xy,expected.tolist(),self.contact_receipt)):
            top = max(float(self.atlas.tile_meshes[index].bounds[1,2]),height)+2.
            hit = query.raycast_closest((float(point[0]),float(point[1]),top),(0.,0.,-1.),
                top-float(self.atlas.tile_meshes[index].bounds[0,2])+1.)
            if not hit.get('hit') or str(hit.get('collision')) != receipt['collision_path']:
                raise RuntimeError(f'Physical terrain ray missed tile {index}: {hit}')
            if str(hit.get('material')) != receipt['material_path']:
                raise RuntimeError(f'Physical terrain material mismatch on tile {index}: {hit.get("material")}')
            position = hit['position']
            actual_height = float(position.z if hasattr(position,'z') else position[2])
            error = abs(actual_height-height)
            if not np.isfinite(error) or error > 2.e-4:
                raise RuntimeError(f'Physical terrain height mismatch on tile {index}: {error}')
            face = int(hit.get('faceIndex',-1))
            if not receipt['face_start'] <= face < receipt['face_end']:
                raise RuntimeError(f'Physical terrain face/material order mismatch on tile {index}: {face}')
            max_error = max(max_error,error)
        self.contact_validation = dict(tiles_checked=len(xy),height_error_max_m=max_error,
            collision_meshes=len(self.contact_receipt),material_subsets=0,cooked_materials_verified=True)
        return dict(self.contact_validation)


class AllTerrainCommand(UniformVelocityCommand):
    def _resample_command(self,env_ids):
        self.vel_command_b[env_ids] = course(self._env).raw_command()[env_ids]
        self.is_standing_env[env_ids] = False
        self.is_heading_env[env_ids] = False

    def _update_command(self):
        self.vel_command_b.copy_(course(self._env).raw_command())


def reset_course(env,env_ids):
    course(env).reset(env_ids)


@configclass
class AllTerrainRewards:
    all_terrain = RewardTermCfg(func=terms.reward,weight=1.)


@configclass
class AllTerrainTerminations:
    all_terrain_failure = TerminationTermCfg(func=terms.failed)
    all_terrain_success = TerminationTermCfg(func=terms.succeeded)
    all_terrain_timeout = TerminationTermCfg(func=terms.timed_out,time_out=True)


@configclass
class AllTerrainEvents:
    joints: EventTermCfg = None
    startup_joints: EventTermCfg = None
    course = EventTermCfg(func=reset_course,mode='reset')
    perception: EventTermCfg = None
    student_heading: EventTermCfg = None
    teacher_heading = EventTermCfg(func=terms.reset_clean_commands,mode='reset')


@configclass
class AllTerrainCurriculum:
    """The single scheduler lives in AllTerrainCourse; disable legacy managers."""


def configure_all_terrain(cfg,settings,algorithm):
    from .all_terrain_config import validate_all_terrain
    cfg.cts_all_terrain = deepcopy(validate_all_terrain(settings))
    cfg.cts_cohort_config = deepcopy(algorithm)
    old = cfg.scene.terrain
    cfg.scene.terrain = AllTerrainImporterCfg(class_type=AllTerrainImporter,prim_path=old.prim_path,
        terrain_type='plane',env_spacing=old.env_spacing or 3.,collision_group=old.collision_group,
        physics_material=deepcopy(old.physics_material),visual_material=deepcopy(old.visual_material),
        debug_vis=False,settings=deepcopy(settings),cohort=deepcopy(algorithm))
    cfg.commands.base_velocity.class_type = AllTerrainCommand
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.debug_vis = False
    cfg.commands.base_velocity.resampling_time_range = (10000.,10000.)
    cfg.rewards = AllTerrainRewards()
    hind = settings.get('hind_clearance')
    if hind and hind['enabled']:
        from . import hind_clearance
        for name, weight in (('hind_step_clearance', 'clearance_weight'),
                             ('hind_riser_contact', 'riser_weight'),
                             ('hind_support_transfer', 'support_weight')):
            setattr(cfg.rewards, name, RewardTermCfg(func=getattr(hind_clearance, name), weight=hind[weight]))
    cfg.terminations = AllTerrainTerminations()
    cfg.curriculum = AllTerrainCurriculum()
    cfg.events = AllTerrainEvents(joints=deepcopy(cfg.events.randomize_reset_joints),
        perception=deepcopy(cfg.events.student_v3_safe_reset),
        student_heading=deepcopy(cfg.events.student_v3_safe_heading_reset))
    if 'startup_stability' in settings:
        from .stability import reset_supported_start_joints
        cfg.events.startup_joints = EventTermCfg(func=reset_supported_start_joints, mode='reset')
    cfg.observations.locomotion_scan.scan.func = terms.locomotion_scan
    cfg.observations.safety.safety.func = terms.teacher_safety
    cfg.observations.teacher.velocity_commands.func = terms.clean_commands
    cfg.observations.teacher_critic.material_friction.func = terms.material_friction
    cfg.student_perception.continuous_delay = True
    cfg.student_perception.validate()
    # Headless Isaac disables PhysX scene queries by default; the cooked
    # collision/material validation must query the actual physics scene.
    cfg.sim.enable_scene_query_support = True
    # Env-ID encoded GPU broadphase expands shared collider bounds to infinity.
    # Explicit collision groups preserve robot isolation without all-to-all
    # robot/tile candidates, and retain continuous single-material GPU contacts.
    cfg.scene.replicate_physics = False
    cfg.scene.filter_collisions = True
    cfg.sim.physx.gpu_collision_stack_size = settings['simulation']['collision_stack_mib']*1024**2
    # Each route has its own timeout; this only bounds the manager's metadata.
    geom,cmd = settings['geometry'],settings['commands']
    cfg.episode_length_s = (2*geom['steps_range'][1]*geom['depth_range'][1]+geom['platform_length_range'][1]
        +geom['route_start']+geom['goal_landing'])/cmd['forward_speed'][0]+cmd['route_time_margin_s']+cmd['stop_s']


def make_env(cfg):
    from rl_training.tasks.manager_based.locomotion.velocity.mdp.cts_env import M20CTSAllTerrainEnv
    return M20CTSAllTerrainEnv(cfg)
