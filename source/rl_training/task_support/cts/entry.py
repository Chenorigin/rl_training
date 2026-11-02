"""CTS command-line, source contracts, resource checks and simulator wiring."""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess

from .configuration import ROOT, TASK, CONFIG_PATH, VERSION, LEGACY_VERSIONS, algorithm_version, default_settings, validate_config


def all_terrain_settings(config):
    value = config.get('all_terrain')
    return value if isinstance(value, dict) and value.get('enabled') else None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--teacher-checkpoint', type=Path)
    source.add_argument('--resume', type=Path)
    source.add_argument('--init-cts', type=Path, help='Transfer CTS weights into a new run with fresh optimizers and current objectives')
    parser.add_argument('--init-student', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--num-envs', type=int)
    parser.add_argument('--max-iterations', type=int, help='Additional iterations, including on resume')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--run-name', default='cts')
    parser.add_argument('--log-root', type=Path)
    parser.add_argument('--min-free-disk-gb', type=float, default=2.)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--preflight-only', action='store_true')
    mode.add_argument('--validate-only', action='store_true')
    parser.add_argument('--validation-steps', type=int, default=24)
    parser.add_argument('--preflight-report', type=Path)
    args = parser.parse_args(argv)
    import os
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        parser.error('CTS currently supports one process/GPU')
    if args.resume and any(x is not None for x in (args.init_student, args.config, args.seed)):
        parser.error('Resume preserves configuration and both models; do not pass initialization/config/seed')
    if args.init_cts and args.init_student is not None:
        parser.error('--init-cts already contains both source actors; do not pass --init-student')
    if not (args.resume or args.init_cts) and args.init_student is None:
        parser.error('A new CTS run requires --teacher-checkpoint and --init-student')
    for name in ('num_envs', 'max_iterations', 'validation_steps'):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error(f'{name} must be positive')
    if args.validate_only and args.validation_steps > 1024:
        parser.error('Inference validation is bounded to 1024 steps')
    if not math.isfinite(args.min_free_disk_gb) or args.min_free_disk_gb <= 0:
        parser.error('min-free-disk-gb must be finite and positive')
    if not args.run_name or Path(args.run_name).name != args.run_name:
        parser.error('run-name must be one directory label')
    return args


def source_hashes():
    from task_support.student_v3_safe.deployment import source_hashes as inherited
    result = inherited()
    task = ROOT / 'source/rl_training/rl_training/tasks/manager_based/locomotion/velocity'
    cfg = task / 'config/wheeled/deeprobotics_m20'
    paths = list(Path(__file__).parent.glob('*.py')) + [CONFIG_PATH,
        ROOT / 'configs/distillation/m20_collision_envelope.json',
        ROOT / default_settings()['sample_stairs']['terrain_xml'],
        ROOT / 'scripts/reinforcement_learning/rsl_rl/train_cts.py',
        ROOT / 'deploy/deploy_mujoco/deploy_cts_student.py',
        ROOT / 'deploy/deploy_mujoco/deploy_mujoco.py',
        ROOT / 'deploy/deploy_mujoco/terrain_loader.py',
        task / 'mdp/cts_env.py', cfg / 'cts_env_cfg.py',
        cfg / 'agents/cts_ppo_cfg.py', cfg / 'cts/__init__.py']
    for path in paths:
        result[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def validate_metadata(meta, *, resume=True):
    from m20_student_v1.contract import validate_contract
    from task_support.stair_quiet_v1.deployment import execution_contract
    from .auxiliary import validate_context
    supported = (VERSION,) if resume else (VERSION, *LEGACY_VERSIONS)
    if meta.get('version') not in supported or meta.get('task') != TASK:
        raise ValueError('Not a compatible M20 CTS checkpoint; older CTS runs cannot resume with changed PPO retention semantics; start a new run')
    validate_config(meta['config'], legacy=meta['version'] != VERSION)
    validate_contract(meta['contract'])
    execution_contract(meta, required=True)
    perception = meta.get('perception')
    if perception != meta.get('environment', {}).get('perception'):
        raise ValueError('CTS environment/perception metadata differ')
    surface = meta['config'].get('map_capture', {}).get('support_surface_visibility', False)
    if perception.get('support_surface_visibility', False) != surface:
        raise ValueError('CTS support-surface sensing differs from saved configuration')
    if meta['version'] in (VERSION, 'm20_cts_action_latent_v11', 'm20_cts_action_latent_v12', 'm20_cts_action_latent_v13', 'm20_cts_action_latent_v14', 'm20_cts_action_latent_v15', 'm20_cts_action_latent_v16', 'm20_cts_action_latent_v17', 'm20_cts_action_latent_v18', 'm20_cts_action_latent_v19', 'm20_cts_action_latent_v20', 'm20_cts_action_latent_v21', 'm20_cts_action_latent_v22', 'm20_cts_action_latent_v23', 'm20_cts_action_latent_v24', 'm20_cts_action_latent_v25'):
        from .perception import validate_perception_contract
        validate_perception_contract(meta['config']['map_delay_mixture'], perception)
        if perception.get('map_capture_padding_m') != meta['config']['map_capture']['padding_m']:
            raise ValueError('CTS map capture and effective sensor configuration differ')
        if all_terrain_settings(meta['config']) is not None and perception.get('continuous_delay') is not True:
            raise ValueError('CTS all-terrain requires continuous map-delay sampling')
    noise = validate_context(meta['config']['auxiliary'], perception)
    auxiliary = meta.get('auxiliary_contract')
    if (not isinstance(auxiliary, dict) or auxiliary.get('version') != 1
            or auxiliary.get('config') != meta['config']['auxiliary']
            or auxiliary.get('sensor_noise_std') != noise
            or 'retention' not in auxiliary):
        raise ValueError('CTS auxiliary configuration/noise contract changed')
    if type(meta.get('num_envs')) is not int or meta['num_envs'] < 2:
        raise ValueError('CTS needs both teacher and student environments')
    if resume and meta.get('source_sha256') != source_hashes():
        raise ValueError('CTS code/environment changed since save; source contract mismatch')
    # Deployment checks the embedded contract only. Retention data are training
    # inputs and must never become an external dependency of the deployed actor.
    if resume and auxiliary != guarded_auxiliary_contract(meta['config'], perception):
        raise ValueError('CTS auxiliary retention/context contract changed')
    return meta


def prepare(args):
    import torch
    from m20_student_v1.model_io import resolve_checkpoint, file_hash
    from m20_student_v1.contract import observation_contract
    from task_support.student_v3_safe.configuration import default_settings as environment_settings
    from task_support.student_v3_safe.teacher import teacher_manifest
    from task_support.stair_quiet_v1.deployment import execution_contract, contract_hash, RUNTIME_SCOPE
    source_path = resolve_checkpoint(args.resume or args.init_cts or args.teacher_checkpoint)
    saved = torch.load(source_path, map_location='cpu', weights_only=True)
    if args.resume:
        meta = deepcopy(validate_metadata(saved.get('cts', {})))
        n = args.num_envs or meta['num_envs']
        if n != meta['num_envs']:
            raise ValueError('CTS resume preserves cohort/environment count')
        teacher, student = saved['initialization']['teacher'], saved['initialization']['student']
        teacher_manifest(teacher)
    else:
        prior = validate_metadata(saved.get('cts', {}), resume=False) if args.init_cts else None
        teacher = (deepcopy(saved['initialization']['teacher']) if prior else
                   {key: saved[key] for key in ('actor_state_dict', 'critic_state_dict', 'm20_training_manifest')})
        manifest = teacher_manifest(teacher)
        student_path = Path(prior['init_student_checkpoint']) if prior else resolve_checkpoint(args.init_student)
        raw_student = saved['initialization']['student'] if prior else torch.load(student_path, map_location='cpu', weights_only=True)
        student = {key: raw_student[key] for key in ('student_state_dict', 'student_v1') if key in raw_student}
        if 'student_state_dict' not in student:
            raise ValueError('--init-student must contain sensor student weights')
        validate_student_source(student, manifest)
        config = default_settings()
        if args.config:
            supplied = json.loads(args.config.read_text())
            if set(supplied) - set(config):
                raise ValueError('Unknown CTS configuration section')
            for key, value in supplied.items():
                if key in ('algorithm', 'auxiliary', 'platform_practice','sample_stairs','legacy_stairs','motion_practice','map_delay_mixture','map_capture','map_recovery','straight_practice','all_terrain','joint_limits'):
                    if not isinstance(value, dict) or set(value) - set(config[key]):
                        raise ValueError(f'Unknown CTS {key} setting')
                    config[key].update(value)
                else:
                    config[key] = value
        if args.seed is not None:
            config['seed'] = args.seed
        validate_config(config)
        n = args.num_envs or (256 if all_terrain_settings(config) is not None else (16 if args.validate_only else 256))
        control = execution_contract({'teacher_manifest': manifest}, required=True)
        env_settings = deepcopy(prior['environment']) if prior else environment_settings()
        env_settings['perception']['zero_delay_probability'] = config['map_delay_mixture']['zero_delay_probability']
        env_settings['perception']['map_capture_padding_m'] = config['map_capture']['padding_m']
        env_settings['perception']['support_surface_visibility'] = config['map_capture'].get('support_surface_visibility', False)
        if all_terrain_settings(config) is not None:
            env_settings['perception']['continuous_delay'] = True
        meta = {'version': VERSION, 'task': TASK, 'num_envs': n, 'config': config,
            'environment': env_settings, 'teacher_manifest': manifest,
            'teacher_checkpoint': prior['teacher_checkpoint'] if prior else str(source_path),
            'teacher_sha256': prior['teacher_sha256'] if prior else file_hash(source_path),
            'init_student_checkpoint': str(student_path),
            'init_student_sha256': prior['init_student_sha256'] if prior else file_hash(student_path),
            'contract': observation_contract(4), 'perception': env_settings['perception'],
            'auxiliary_contract': guarded_auxiliary_contract(config, env_settings['perception']),
            'command_feedback': {'version': 'legacy_v1_v3_5', 'tuning': manifest['stair_tuning']},
            'quiet_execution_control': control, 'quiet_execution_control_sha256': contract_hash(control),
            'quiet_runtime_scope': RUNTIME_SCOPE, 'source_sha256': source_hashes(),
            'student_inputs': {'policy': 789, 'motion_control': 3},
            'architecture': ('CTS shared residual decoder, unnormalized action latent16, Safe '
                             + ('trainable base' if config['algorithm'].get('train_teacher_base', False) else 'frozen base')),
            'resume_semantics': 'Both optimizers/models, auxiliary data/noise contract, counters, cohorts and completed-course ledger; physics/sensor history reset'}
        if prior:
            meta['warm_start'] = dict(checkpoint=str(source_path), sha256=file_hash(source_path),
                previous_updates=saved['algorithm']['num_updates'],
                transfer='all model weights including shared decoder/critic; optimizer, iteration, RNG, cohorts and course ledger reset')
        validate_metadata(meta)
    args.log_root = (args.log_root or ROOT / ('logs/evaluation/cts_validation' if args.validate_only
                                           else 'logs/rsl_rl/deeprobotics_m20_cts')).expanduser().resolve()
    return saved if args.resume or args.init_cts else None, teacher, student, meta


def initialize_cts_weights(algorithm, saved):
    """Explicit weight transfer, never pretend old Adam/course state resumed.

    Reuse the atomic restore validator against a fresh optimizer/cohort/RNG
    state, replacing only model tensors. Frozen sources and configured decoder
    bounds must still match exactly; malformed/nonfinite tensors fail closed.
    """
    meta, previous = saved['cts'], saved['algorithm']
    expected = algorithm_version(meta['version'])
    if (previous.get('cts_version') != expected
            or previous.get('algorithm_config') != meta['config']['algorithm']
            or previous.get('auxiliary_contract') != meta['auxiliary_contract']):
        raise ValueError('CTS warm-start model/configuration contract mismatch')
    fresh = algorithm.save()
    from .model import transfer_state
    fresh['models_state_dict'] = transfer_state(algorithm.models, previous['models_state_dict'])
    algorithm.load(fresh)


def validate_student_source(student, manifest):
    from m20_student_v1.contract import validate_contract
    from task_support.student_v3_safe.deployment import validate_provenance
    provenance = student.get('student_v1', {}).get('provenance', {})
    validate_provenance(provenance)
    validate_contract(provenance['contract'])
    cfg = provenance.get('training_config', {}).get('student', {})
    if (cfg.get('activation') != 'elu' or cfg.get('obs_normalization') is not False
            or cfg.get('hidden_dims') != [512,256,128]
            or provenance['command_feedback']['tuning'] != manifest['stair_tuning']):
        raise ValueError('CTS student activation/observation/command ABI differs from the source teacher')


def auxiliary_memory_budget(config):
    """Reserve data copies and bounded student/decoder autograd workspaces."""
    from .auxiliary import retention_bytes
    auxiliary = config['auxiliary']
    data_bytes = retention_bytes(auxiliary)
    stability_batches = int(auxiliary['temporal_loss_weight'] > 0) + int(auxiliary['spatial_loss_weight'] > 0)
    rows = 2*auxiliary['stability_batch_size']*stability_batches
    if auxiliary['retention_loss_weight'] > 0:
        rows += auxiliary['retention_batch_size']
    if config.get('straight_practice'):
        rows += config['straight_practice']['batch_size']
    from .wall_stability import settings as wall_settings
    wall = wall_settings(config.get('all_terrain'))
    if wall is not None:
        # Clean/degraded pair plus the hard-idle student minibatch.
        rows += 2*wall['batch_size'] + config['algorithm']['student_batch_size']
    # Current source ABI: policy789, command3, action16 and student512/256/128.
    # Eight float32 copies cover paired forwards, autograd and backend workspace;
    # three dataset copies cover deserialization, validation and device transfer.
    features = 789+3+16+512+256+128+sum(config['algorithm']['decoder_hidden_dims'])+16
    temporary_bytes = rows*features*4*8
    return {'retention_data_bytes': data_bytes,
            'auxiliary_temporary_bytes': temporary_bytes,
            'auxiliary_reserve_bytes': 3*data_bytes+temporary_bytes}


def require_auxiliary_headroom(host, budget):
    if host.total-host.available+2*1024**3+budget['auxiliary_reserve_bytes'] > .85*host.total:
        raise RuntimeError('Insufficient host headroom for CTS CPU preflight')


def guarded_auxiliary_contract(config, perception):
    """Validate datasets only after reserving CPU deserialization headroom."""
    import psutil
    from .auxiliary import context_contract
    require_auxiliary_headroom(psutil.virtual_memory(), auxiliary_memory_budget(config))
    return context_contract(config['auxiliary'], perception)


def validate_live_auxiliary(algorithm, meta):
    if algorithm.auxiliary.contract != meta['auxiliary_contract']:
        raise ValueError('CTS live auxiliary contract differs from checkpoint metadata')


def all_terrain_memory_budget(cfg, host):
    """Bound atlas construction first, then account for the actual mesh arrays."""
    from .all_terrain_geometry import DEFAULTS, build_atlas
    from .all_terrain_config import COURSES
    geometry = {**DEFAULTS, **cfg['geometry']}
    tiles = len(COURSES)*geometry['geometry_levels']*geometry['samples_per_level']
    walls = cfg.get('wall_practice')
    if walls and walls['enabled']:
        from .wall_practice import KINDS
        tiles += len(KINDS)*geometry['geometry_levels']*walls['samples_per_kind_level']
    nx = math.ceil(2*geometry['field_half_length']/geometry['grid_resolution'])+2
    ny = math.ceil(geometry['tile_width']/geometry['grid_resolution'])+2
    boxes = 2*geometry['steps_range'][1]+geometry['obstacle_count_range'][1]+8
    enclosed = cfg.get('stair_walls',{}).get('enabled',False)
    if enclosed:boxes += 2*(2*geometry['steps_range'][1]+3)
    construction_bound = tiles*(nx*ny*96+boxes*480)*6
    if host.total-host.available+construction_bound+2*1024**3 > .85*host.total:
        raise RuntimeError('Insufficient host headroom to construct the all-terrain atlas')
    atlas = build_atlas(cfg['geometry'],cfg.get('wall_practice'),cfg.get('obstacle_response'),cfg.get('stair_walls'))
    payload = atlas.mesh.vertices.nbytes+atlas.mesh.faces.nbytes
    arrays = sum(getattr(atlas, name).nbytes for name in
                 ('origins','course_ids','geometry_levels','spawns','goals','bounds','local_bounds','goal_local'))
    arrays += sum(a.nbytes for grid in atlas.grids for a in grid)
    arrays += sum(a.nbytes for a in atlas.rectangles)
    # The surface query pads each tile's height and obstacle tables on device.
    padded = tiles*max(len(g[0]) for g in atlas.grids)*max(len(g[1]) for g in atlas.grids)*4
    padded += tiles*max(1, max(len(r) for r in atlas.rectangles))*5*4
    if enclosed:
        # Separate traversable-support and complete ray surfaces, plus wall ledger/envelopes.
        wall_count=max(len(r.get('stair_wall_rectangles',[])) for r in atlas.records)
        arrays += sum(len(r.get('stair_wall_rectangles',[]))*5*8 for r in atlas.records)
        padded = 2*padded+tiles*wall_count*5*4*3
    # Eight mesh copies cover per-tile/concatenated CPU meshes, USD/PhysX and
    # raycasting conversion buffers; three surface copies include CPU and GPU.
    reserve = payload*8+(arrays+padded)*3
    records = atlas.records
    info = {'tiles':tiles, 'vertices':len(atlas.mesh.vertices), 'faces':len(atlas.mesh.faces),
            'mesh_payload_bytes':payload, 'surface_arrays_bytes':arrays,
            'surface_padded_bytes':padded, 'construction_upper_bound_bytes':construction_bound,
            'records_sha256':hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest(),
            'sampling':'continuous parameter draws at atlas construction; immutable episode geometry',
            'tiles_per_course':{name:sum(r['course'] == name for r in records) for name in COURSES},
            'stair_height_range_m':[min(r['height'] for r in records if r['course_id'] < 3),
                                    max(r['height'] for r in records if r['course_id'] < 3)],
            'stair_depth_range_m':[min(r['depth'] for r in records if r['course_id'] < 3),
                                   max(r['depth'] for r in records if r['course_id'] < 3)]}
    if enclosed:
        from .stair_walls import LAYOUTS
        info['stair_wall_tiles']={name:{layout:sum(r['course']==name and r.get('stair_wall_layout')==layout for r in records)
            for layout in LAYOUTS} for name in COURSES[:3]}
    return reserve, info


def preflight(args, teacher, student, meta, resume_saved=None):
    import psutil
    import torch
    from .model import build_models
    from .algorithm import CTSAlgorithm
    from m20_student_v0.output_storage import require_disk_space
    host = psutil.virtual_memory()
    cfg, n = meta['config'], meta['num_envs']
    auxiliary_budget = auxiliary_memory_budget(cfg)
    # CPU checkpoint/model inspection is bounded independently of the simulator.
    require_auxiliary_headroom(host, auxiliary_budget)
    from .practice import cohort_mask, practice_mask
    cohorts = cohort_mask(n,cfg['algorithm']['teacher_fraction'],cfg['algorithm']['cohort_seed'])
    all_terrain = all_terrain_settings(cfg)
    course_allocation = None
    if all_terrain is not None:
        from .all_terrain_curriculum import allocate_courses
        from .all_terrain_config import COURSES
        courses = allocate_courses(cohorts,all_terrain['fractions'],cfg['algorithm']['cohort_seed'])
        counts = [[int(((courses == i) & cohort).sum()) for i in range(len(COURSES))] for cohort in (cohorts,~cohorts)]
        if any(count == 0 for row in counts for count in row):
            raise ValueError('CTS all-terrain needs every course in both cohorts; increase num_envs')
        course_allocation = {name:{'teacher':counts[0][i],'student':counts[1][i],
                                  'total':counts[0][i]+counts[1][i],
                                  'target_fraction':all_terrain['fractions'][i]} for i,name in enumerate(COURSES)}
        platform, straight = courses == 9, courses == 8
        motion = (courses == 3) | straight
        stairs_count = int((courses < 3).sum())
    else:
        platform = practice_mask(cohorts,cfg['platform_practice']['fraction'],cfg['algorithm']['cohort_seed']+71)
        motion = practice_mask(cohorts,cfg['motion_practice']['fraction'],cfg['algorithm']['cohort_seed']+113,platform)
        straight = practice_mask(cohorts,cfg['straight_practice']['fraction'],cfg['algorithm']['cohort_seed']+211,~motion)
        if any(not (motion & ~straight & c).any() for c in (cohorts,~cohorts)):
            raise ValueError('CTS needs ordinary and straight motion in both cohorts; increase num_envs')
        if any(not (cohort & ~(platform | motion)).any() for cohort in (cohorts,~cohorts)):
            raise ValueError('CTS environment count must leave stairs in both teacher and student cohorts')
        stairs_count = int((~(platform | motion)).sum())
    torch.set_num_threads(1)
    torch.manual_seed(meta['config']['seed'])
    models = build_models(teacher, student, cfg['algorithm'], device='cpu', all_terrain_cfg=all_terrain)
    # Tiny algorithm validates optimizer configuration and complete resume state,
    # without allocating the requested simulator's rollout on CPU.
    alg = CTSAlgorithm(models, cfg['algorithm'], n, cfg['num_steps_per_env'], 'cpu',
                       auxiliary_cfg=cfg['auxiliary'], perception=meta['perception'], straight_cfg=meta['config']['straight_practice'],
                       all_terrain_cfg=all_terrain, joint_limit_cfg=cfg.get('joint_limits'))
    validate_live_auxiliary(alg, meta)
    if alg.auxiliary.memory_bytes > auxiliary_budget['retention_data_bytes']:
        raise RuntimeError('CTS auxiliary allocation exceeded its preflight estimate')
    if resume_saved is not None:
        if args.resume:
            alg.load(resume_saved['algorithm'])
        else:
            initialize_cts_weights(alg, resume_saved)
    from .runner import parity_check
    parity = parity_check(models, teacher, student, initialized=resume_saved is None)
    parameter_bytes = sum(p.numel()*p.element_size() for p in models.parameters())
    buffer_bytes = sum(p.numel()*p.element_size() for p in models.buffers())
    # Live + reference + gradients + Adam moments + activation margin.
    model_reserve = 8*(parameter_bytes+buffer_bytes)
    rollout = n*cfg['num_steps_per_env']*(789+244+3+187+48+285+16*4+8)*4
    # Keep complete source states for simulator-free, source-file-free reconstruction.
    source_bytes = sum(v.numel()*v.element_size() for item in (teacher, student)
                       for key, state in item.items() if key.endswith('state_dict') for v in state.values())
    checkpoint = math.ceil(1.5*(model_reserve+source_bytes)+1024**2)
    iterations = args.max_iterations or cfg['max_iterations']
    count = math.ceil(iterations/cfg['save_interval'])+2
    reserve = math.ceil(args.min_free_disk_gb*1024**3)
    required = reserve + (checkpoint*2 if args.validate_only else count*checkpoint+iterations*8192)
    disk = require_disk_space(args.log_root, required)
    atlas_report = None
    if all_terrain is not None:
        mesh_bytes, atlas_report = all_terrain_memory_budget(all_terrain, host)
        platform_mesh_bytes = motion_mesh_bytes = straight_mesh_bytes = 0
        legacy_geometry_report = []
        sample_report = {'enabled':False, 'reason':'replaced by the all-terrain route curriculum'}
    else:
        from dataclasses import replace
        from task_support.stair_quiet_v1_safe.settings import SafeSettings
        from task_support.student_v3_safe.terrain_geometry import course_mesh
        from .terrain import geometries
        terrain = meta['environment']['training']['terrain_course']
        settings = replace(SafeSettings(**meta['teacher_manifest']['noise_refinement']['contract']['environment']['stair_tail']),
                           tile_length=terrain['tile_length'])
        from .sample_stairs import sample_mesh, scene as sample_scene
        meshes = [sample_mesh(settings,cfg['sample_stairs']['terrain_xml']) if g.family == 'round_trip'
                  else course_mesh(g, settings) for g in geometries(terrain,cfg['legacy_stairs'])]
        mesh_bytes = sum(m.vertices.nbytes+m.faces.nbytes for m in meshes)*5*len(settings.widths_m)*8
        # Basis for one closed box per row/contact column, eight mesh copies.
        box_mesh_bytes = len(settings.widths_m)*len(meshes)*5*(8*3*8+12*3*8)*5*8
        # Two boxes (raised slab + floor) per height/row/column, same reserve basis.
        platform_mesh_bytes = 2*box_mesh_bytes*len(cfg['platform_practice']['height_bands'])*cfg['platform_practice']['samples_per_band']
        mesh_bytes += platform_mesh_bytes
        from .motion import motion_patch_mesh
        from types import SimpleNamespace
        motion_mesh,_ = motion_patch_mesh(SimpleNamespace(num_rows=len(settings.widths_m),num_cols=len(meshes)*5,
                                                        size=(settings.tile_length,settings.tile_width)),cfg['motion_practice'])
        motion_mesh_bytes = (motion_mesh.vertices.nbytes+motion_mesh.faces.nbytes)*8
        from .straight import straight_patch_mesh
        straight_mesh,_ = straight_patch_mesh(SimpleNamespace(num_rows=len(settings.widths_m),num_cols=len(meshes)*5,
            size=(settings.tile_length,settings.tile_width)),cfg['motion_practice'],cfg['straight_practice'])
        straight_mesh_bytes = (straight_mesh.vertices.nbytes+straight_mesh.faces.nbytes)*8
        mesh_bytes += straight_mesh_bytes
        mesh_bytes += motion_mesh_bytes
        legacy_geometry_report = [{'name':g.name,'steps':g.steps,'height_m':g.height,'tread_depth_m':g.depth,
                                  'total_height_m':g.steps*g.height}
                                  for g in geometries(terrain,cfg['legacy_stairs']) if g.family == 'legacy']
        sample_report = {'xml':cfg['sample_stairs']['terrain_xml'],
            'xml_sha256':hashlib.sha256((ROOT/cfg['sample_stairs']['terrain_xml']).read_bytes()).hexdigest(),
            'native_initial_centreline_treads':sample_scene(cfg['sample_stairs']['terrain_xml'])[1],
            'conservative_corridor_width_m':sample_scene(cfg['sample_stairs']['terrain_xml'])[2],
            'collision_dynamics':'static initial native geometry; original contact/robot/sensor domain randomization',
            'reward':deepcopy(cfg['sample_stairs'])}
    from task_support.student_v1.capture_budget import capture_memory_budget
    capture_budget = capture_memory_budget(meta['perception'],n)
    from .map_recovery import memory_reserve
    recovery_bytes = memory_reserve(cfg.get('map_recovery'), meta['perception'], n)
    # Normals, short recast XYZ/directions/results and visibility work, with
    # 2x temporary headroom. The base sensor/queue budget already covers hits.
    from task_support.student_v1.perception import capture_grid_shape
    surface_rows, surface_cols = capture_grid_shape(meta['perception'].get('map_capture_padding_m', 0.))
    surface_bytes = (n * surface_rows * surface_cols * 256
                     if meta['perception'].get('support_surface_visibility', False) else 0)
    # Extra full-rollout clean action targets, transient clone, and pre-step bool tags.
    straight_target_bytes = n*cfg['num_steps_per_env']*(16*4*3+1)
    course_tag_bytes = n*cfg['num_steps_per_env']*9*3
    # Dense atlas contact generation needs a separately sized PhysX stack. Do
    # not hide this explicit allocation inside the baseline simulator reserve.
    collision_stack_bytes = (all_terrain['simulation']['collision_stack_mib']*1024**2
                             if all_terrain is not None else 0)
    allowance = (8*1024**3+n*4*1024**2+rollout*3+model_reserve+mesh_bytes
                 +auxiliary_budget['auxiliary_reserve_bytes']+capture_budget['extra_capture_reserve_bytes']+surface_bytes
                 +straight_target_bytes+course_tag_bytes+collision_stack_bytes+recovery_bytes)
    if host.total-host.available+allowance > .85*host.total:
        raise RuntimeError('Insufficient whole-host headroom for CTS')
    report = {'task': TASK, 'num_envs': n, 'device': args.device, **parity,
        'teacher_trainable_parameters': sum(p.numel() for p in models.teacher.parameters() if p.requires_grad),
        'ppo_parameter_groups': [dict(name=group.get('name', 'legacy_ppo'), lr=group['lr'],
                                     parameters=sum(p.numel() for p in group['params']))
                                 for group in alg.optimizer.param_groups],
        'host_available_bytes': host.available, 'host_total_bytes': host.total,
        'swap_used_bytes': psutil.swap_memory().used, 'load_average': list(__import__('os').getloadavg()),
        'disk_free_bytes': disk, 'planned_increment_bytes': allowance,
        'support_surface_reserve_bytes': surface_bytes,
        'checkpoint_estimate_bytes': checkpoint, 'required_free_bytes': required, 'min_free_disk_bytes': reserve,
        'basis': '8 GiB simulator + configured additional PhysX collision stack + 4 MiB/env + 3x float32 rollout tensors + 8x model + 8x terrain mesh copies + 3x retention data + 8x auxiliary batch activations + expanded source-map queue/work/ray reserve; 85% total occupancy cap',
        **capture_budget, 'rollout_bytes': rollout, 'terrain_mesh_reserve_bytes': mesh_bytes,
        'platform_mesh_reserve_bytes': platform_mesh_bytes,
        'motion_mesh_reserve_bytes': motion_mesh_bytes, 'straight_mesh_reserve_bytes':straight_mesh_bytes,
        'straight_target_reserve_bytes':straight_target_bytes, 'course_tag_reserve_bytes':course_tag_bytes,
        'collision_stack_reserve_bytes':collision_stack_bytes,
        'map_recovery_reserve_bytes': recovery_bytes,
        'map_recovery': deepcopy(cfg.get('map_recovery')),
        'practice_allocation': {'platform':int(platform.sum()),'motion':int(motion.sum()),
                                'straight':int(straight.sum()),'ordinary_motion':int((motion & ~straight).sum()),
                                'straight_teacher':int((straight & cohorts).sum()),'straight_student':int((straight & ~cohorts).sum()),
                                'motion_teacher':int((motion & cohorts).sum()),'motion_student':int((motion & ~cohorts).sum()),
                                'stairs':stairs_count},
        'legacy_stairs':legacy_geometry_report, 'sample_stairs':sample_report,
        'all_terrain_allocation':course_allocation, 'all_terrain_atlas':atlas_report,
        **auxiliary_budget, 'auxiliary_memory_bytes': alg.auxiliary.memory_bytes,
        'auxiliary_contract': deepcopy(alg.auxiliary.contract),
        'trained': False, 'optimizer_updates': 0, 'simulator_started': False}
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        free, total = torch.cuda.mem_get_info(torch.device(args.device))
        if total-free+allowance > .85*total:
            raise RuntimeError('Insufficient whole-GPU headroom for CTS')
        report.update(gpu_free_bytes=free, gpu_total_bytes=total)
        report['nvidia_smi'] = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu', '--format=csv'], text=True).strip()
    if args.preflight_report:
        args.preflight_report.parent.mkdir(parents=True, exist_ok=True)
        args.preflight_report.write_text(json.dumps(report, indent=2)+'\n')
    print('CTS_PREFLIGHT='+json.dumps(report), flush=True)
    return report


def configure_environment(meta, device):
    from isaaclab_tasks.utils import parse_env_cfg
    from m20_student_v1.configuration import StudentV1PerceptionCfg
    from task_support.stair_quiet_v1.landing import LandingSettings
    from task_support.stair_quiet_v1_safe.settings import SafeSettings
    from task_support.student_v3_safe.terrain_importer import configure_course
    cfg = parse_env_cfg(TASK, device=device, num_envs=meta['num_envs'])
    settings, manifest = meta['environment'], meta['teacher_manifest']
    cfg.student_perception = StudentV1PerceptionCfg(**settings['perception'])
    cfg.student_perception.validate()
    cfg.student_history_steps = 4
    cfg.student_idle_practice = deepcopy(settings['training']['idle_practice'])
    cfg.student_diagonal_reward = deepcopy(settings['training']['diagonal_reward'])
    cfg.rewards.student_diagonal_alternation.weight = cfg.student_diagonal_reward['weight']
    cfg.motion_teacher_tuning = deepcopy(manifest['stair_tuning'])
    cfg.seed = meta['config']['seed']
    cfg.landing_control = LandingSettings(**meta['quiet_execution_control']['settings'])
    cfg.stair_tail = SafeSettings(**manifest['noise_refinement']['contract']['environment']['stair_tail'])
    for case in cfg.scene.terrain.terrain_generator.sub_terrains.values():
        case.settings = cfg.stair_tail
    configure_course(cfg, settings['training']['terrain_course'])
    from .practice_isaac import configure_practice
    configure_practice(cfg, meta['config']['algorithm'],meta['config']['platform_practice'],meta['config']['sample_stairs'],meta['config']['legacy_stairs'],meta['config']['motion_practice'],meta['config']['map_delay_mixture'],meta['config']['straight_practice'])
    from task_support.student_v1.capture_isaac import configure_capture
    configure_capture(cfg)
    all_terrain = all_terrain_settings(meta['config'])
    if all_terrain is not None:
        from .all_terrain_isaac import configure_all_terrain
        configure_all_terrain(cfg, all_terrain, meta['config']['algorithm'])
    if 'joint_limits' in meta['config']:
        from .joint_limits_isaac import configure as configure_limits
        configure_limits(cfg, meta['config']['joint_limits'])
    if 'map_recovery' in meta['config']:
        from .map_recovery import configure as configure_recovery
        configure_recovery(cfg, meta['config']['map_recovery'])
    cfg.commands.base_velocity.debug_vis = False
    return cfg


def main(argv=None):
    args = parse_args(argv)
    saved, teacher, student, meta = prepare(args)
    report = preflight(args, teacher, student, meta, saved)
    if args.preflight_only:
        return report
    from isaaclab.app import AppLauncher
    app = AppLauncher({'device': args.device, 'device_explicit': True, 'headless': args.headless,
                       'enable_cameras': False, 'livestream': 0, 'xr': False}).app
    env, failed = None, False
    try:
        import gymnasium as gym
        import importlib
        importlib.import_module('rl_training.tasks')
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from task_support.student_v3_safe.environment import initialize_course
        from .model import build_models
        from .algorithm import CTSAlgorithm
        from .runner import CTSRunner
        cfg = configure_environment(meta, args.device)
        if args.validate_only:
            cfg.scene.terrain.visual_material = None
            cfg.scene.sky_light = None
        all_terrain = all_terrain_settings(meta.get('config', {}))
        if all_terrain is not None:
            from .all_terrain_isaac import make_env
            env = RslRlVecEnvWrapper(make_env(cfg), clip_actions=100.)
        else:
            env = RslRlVecEnvWrapper(gym.make(TASK, cfg=cfg), clip_actions=100.)
        models = build_models(teacher, student, meta['config']['algorithm'], args.device, all_terrain_cfg=all_terrain)
        alg = CTSAlgorithm(models, meta['config']['algorithm'], meta['num_envs'],
                           meta['config']['num_steps_per_env'], args.device,
                           auxiliary_cfg=meta['config']['auxiliary'], perception=meta['perception'], straight_cfg=meta['config']['straight_practice'],
                           all_terrain_cfg=all_terrain, joint_limit_cfg=meta['config'].get('joint_limits'))
        validate_live_auxiliary(alg, meta)
        import torch
        environment_cohorts = (env.unwrapped.cts_all_terrain.teacher if all_terrain is not None else env.unwrapped.cts_motion.teacher)
        if not torch.equal(environment_cohorts, alg.teacher_mask):
            raise RuntimeError('CTS environment and PPO cohort identities differ')
        directory = args.log_root / (datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')+'_'+args.run_name)
        directory.mkdir(parents=True, exist_ok=False)
        runner = CTSRunner(env, alg, meta, {'teacher': teacher, 'student': student}, directory, report)
        if args.resume:
            runner.load(saved)
        else:
            if saved is not None:
                initialize_cts_weights(alg, saved)
            if all_terrain is not None:
                with torch.inference_mode():
                    env.reset()
            else:
                initialize_course(env.unwrapped, meta['teacher_manifest'])
        if all_terrain is not None:
            env.unwrapped.scene.terrain.validate_contacts()
        (directory/'resource_preflight.json').write_text(json.dumps(report, indent=2)+'\n')
        (directory/'provenance.json').write_text(json.dumps(meta, indent=2)+'\n')
        practice = getattr(env.unwrapped.cts_all_terrain,'stair_hold',None) if all_terrain_settings(meta['config']) is not None else None
        if practice is not None:practice.validate_initial()
        wall_hold = getattr(env.unwrapped.cts_all_terrain,'wall_hold',None) if all_terrain is not None else None
        if wall_hold is not None:wall_hold.validate_initial()
        print('CTS_RUN_DIR='+str(directory), flush=True)
        if args.validate_only:
            return runner.validate(args.validation_steps)
        runner.learn(args.max_iterations or meta['config']['max_iterations'])
    except BaseException:
        import traceback
        import sys
        failed = True
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        # Kit may terminate with status zero *inside* close(), so a finally
        # after close is too late. The OS releases this process's CUDA context.
        import os
        os._exit(1)
    finally:
        try:
            if env is not None:
                env.close()
            app.close()
        finally:
            if failed:
                import os
                os._exit(1)  # Kit can otherwise turn an uncaught error into exit 0.
