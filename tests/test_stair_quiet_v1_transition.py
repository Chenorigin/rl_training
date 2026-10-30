"""Transition routing, protocol, physical-risk counterexamples and strict state."""
from pathlib import Path
import sys
from types import SimpleNamespace as NS
from contextlib import redirect_stdout
from io import StringIO
from copy import deepcopy
import torch
import pytest
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"source/rl_training"))
sys.path.insert(0, str(ROOT/"tests"))
from test_idle_stop_motion_v1 import ObservationEnv, train_cfg
from test_stair_quiet_v1_width_mechanics import sample
from task_support.stair_quiet_v1_transition.settings import TransitionSettings, cases
from task_support.stair_quiet_v1_transition.protocol import EntryProtocol
from task_support.stair_quiet_v1_transition.signals import TransitionTracker
from task_support.stair_quiet_v1_transition.course import course_state, validate_transition_counts
from task_support.stair_quiet_v1_transition import runner as runtime
from task_support.stair_quiet_v1_safe.actor import SafeResidualActor
from task_support.stair_quiet_v1.landing import LandingSettings
from rsl_rl.models import MLPModel

torch.set_num_threads(1)


@pytest.fixture
def make_runner():
    def make():
        env = ObservationEnv()
        env.obs["locomotion_scan"] = env.obs["policy"][:, 57:].clone()
        env.obs["safety"] = torch.zeros(4, 48)
        env.cfg.stair_tail, env.cfg.landing_control = TransitionSettings(), LandingSettings()
        env.stair_quiet_geometry, env.tail_wheel_material_receipt = {}, {}
        env.stair_quiet_landing = NS(contract=lambda: {})
        env.scene = NS(terrain=NS(contact_receipt=[], terrain_types=torch.zeros(4, dtype=torch.long),
                                  terrain_levels=torch.zeros(4, dtype=torch.long)))
        env.resets = 0
        def reset():
            env.resets += 1
        env.reset = reset
        cfg = train_cfg()
        cfg.update(descent_guide_iters=0, descent_guide_decay_iters=0)
        with redirect_stdout(StringIO()):
            r = runtime.StairQuietTransitionRunner(env, cfg, device="cpu")
        r.logger.writer = None
        return r
    return make


def test_safe18400_complete_migration_and_35_case_targets(make_runner):
    r = make_runner()
    r.initialize_task({})
    saved = torch.load(ROOT/runtime.DEFAULT_START, weights_only=True, map_location="cpu")
    for kind in ("actor", "critic"):
        assert all(torch.equal(v, getattr(r.alg, kind).state_dict()[k]) for k, v in saved[kind+"_state_dict"].items())
    assert r._initialization["source_iteration"] == 18400
    assert r._initialization["source_task"] == 'stair_quiet_v1_safe'
    assert r.alg.actor.safety_mlp[-1].weight.count_nonzero() > 0
    assert all(torch.equal(v,r.alg.reference_actor.state_dict()[k]) for k,v in saved['preservation_reference_actor'].items())
    assert not r.alg.optimizer.state and r.alg.guard_updates == 0
    state = course_state(r.env.unwrapped)
    assert state["width_statistics"]["episodes"].shape == (10, 35)
    assert state["transition_statistics"]["episodes"].shape == (4, 35)
    levels = saved["m20_training_manifest"]["width_curriculum"]["level"]
    pool=[v for i,v in enumerate(levels) if i%4!=0]
    assert state["width_curriculum"].level.tolist() == [7,pool[0],pool[len(pool)//3],pool[2*len(pool)//3]]
    assert not state['width_curriculum'].success_streak.any()
    assert r.env.tail_stage==saved['m20_training_manifest']['tail_curriculum']['stage']==2
    assert not state['episodes'].any() and not state['width_statistics']['episodes'].any()
    assert all(not p.requires_grad for p in r.alg.actor.mlp.parameters())
    assert all(not p.requires_grad for p in r.alg.actor.distribution.parameters())
    r.alg.actor(r.env.obs).square().mean().backward()
    assert r.alg.actor.safety_mlp[-1].weight.grad.abs().sum()>0
    assert all(p.grad is None for p in r.alg.actor.mlp.parameters())


def test_width_cannot_replace_safe_initialization(make_runner):
    from task_support.stair_quiet_v1_safe.runner import DEFAULT_START as width_source
    with pytest.raises(ValueError,match='requires an intact current Safe'):
        make_runner().initialize_task(dict(init_actor_from=str(ROOT/width_source)))


@pytest.mark.parametrize('kind',['missing_safety','nonfinite_safety','changed_reference','source_hash','invalid_stage','invalid_width'])
def test_incomplete_or_changed_safe_source_is_rejected(make_runner,tmp_path,kind):
    saved=torch.load(ROOT/runtime.DEFAULT_START,weights_only=True,map_location='cpu')
    if kind=='missing_safety':del saved['actor_state_dict']['safety_mlp.4.weight']
    elif kind=='nonfinite_safety':saved['actor_state_dict']['safety_mlp.4.weight'][0,0]=float('nan')
    elif kind=='changed_reference':saved['preservation_reference_actor']['mlp.0.weight'][0,0]+=1
    elif kind=='source_hash':saved['m20_training_manifest']['noise_refinement']['source_sha256']={}
    elif kind=='invalid_stage':saved['m20_training_manifest']['tail_curriculum']['stage']=3
    else:saved['m20_training_manifest']['width_curriculum']['level'][1]=99
    path=tmp_path/'bad_safe.pt';torch.save(saved,path)
    with pytest.raises(ValueError):make_runner().initialize_task(dict(init_actor_from=str(path)))


def test_scan_route_changes_zero_only_and_safety_raw_is_preserved(make_runner):
    r = make_runner();r.initialize_task({})
    actor, obs = r.alg.actor, r.env.obs.clone()
    kwargs = dict(r._actor_model_config);kwargs.pop("class_name")
    old = SafeResidualActor(obs, {"actor": ["policy"]}, "actor", 16, **kwargs)
    old.load_state_dict(actor.state_dict())
    obs["motion_control"][:] = .5
    obs["locomotion_scan"] += .4
    assert torch.equal(actor(obs), old(obs))
    obs["motion_control"][:] = 0
    routed = {"policy": torch.cat((obs["policy"][:,:57],obs["locomotion_scan"]),-1)}
    assert torch.equal(actor(obs), MLPModel.forward(actor, routed))
    assert not torch.equal(actor(obs), old(obs))  # Original routing must fail this counterexample.
    before = actor(obs).clone()
    with torch.no_grad():actor.safety_mlp[-1].bias.add_(1)
    assert torch.equal(actor(obs), before)  # Stop residual gate remains closed.
    obs["locomotion_scan"] = obs["policy"][:,57:].clone()
    assert torch.equal(actor(obs), old(obs))  # Original flat scan remains identical.


def test_protocol_idempotence_consumed_pause_and_irreversible_top_stop():
    s = TransitionSettings();p = EntryProtocol(4,"cpu",s);ids=torch.arange(4)
    p.reset(ids,torch.zeros(4,dtype=torch.long));p.mode[:]=1;p.trigger[:]=.4;p.pause[:]=1;p.ramp[:]=.5
    gap=torch.full((4,),.3);crossed=torch.zeros(4,dtype=torch.bool);speed=torch.full((4,),.6)
    assert not p.command(ids,10.,gap,crossed,speed).any()
    assert not p.command(ids,10.,gap,crossed,speed).any()
    assert not p.command(ids,10.5,gap,crossed,speed).any()
    torch.testing.assert_close(p.command(ids,11.25,gap,crossed,speed),torch.full((4,),.3))
    torch.testing.assert_close(p.command(ids,12.,gap,crossed,speed),speed)
    crossed[0]=True
    assert p.command(ids,13.,gap,crossed,speed)[0]==0
    crossed[:]=False
    assert p.command(ids,14.,gap+1,crossed,speed)[0]==0
    snapshot=deepcopy(vars(p));p.reset(torch.tensor([2]),torch.tensor([3]))
    assert p.stopped[0] and p.phase[1]==snapshot['phase'][1]
    assert p.started_at[1]==snapshot['started_at'][1]


def test_protocols_include_direct_without_mandatory_stop_and_continuous_ramp():
    p=EntryProtocol(4,'cpu',TransitionSettings());ids=torch.arange(4);p.reset(ids,torch.zeros(4,dtype=torch.long))
    p.trigger[:]=.4;p.ramp[:]=1
    g=torch.ones(4);c=torch.zeros(4,dtype=torch.bool);s=torch.full((4,),.6)
    out=p.command(ids,0.,g,c,s)
    assert out.tolist()==pytest.approx([.6,.6,.3,.6])
    g[:]=.3;out=p.command(ids,1.,g,c,s)
    assert out[0]==s[0] and out[3]==s[3] and out[2]==pytest.approx(.3)
    assert p.command(ids,1.5,g,c,s)[2]==pytest.approx(.45)


def test_riser_work_is_proxy_pre_solve_gated_and_reset_safe():
    t=TransitionTracker(2,'cpu',TransitionSettings());d=sample(n=2,x=.94,speed=1.,force=20.)
    d['forces'][...,0]=-500.;d['position'][...,2]=.20
    t.update(closed_riser=torch.ones(2,dtype=torch.bool),**d)
    assert t.riser_work_proxy_J.sum()==0  # No fabricated pre-step velocity on first frame.
    t.update(closed_riser=torch.ones(2,dtype=torch.bool),**d)
    assert (t.riser_work_proxy_J>0).all()
    before=t.riser_work_proxy_J[1].clone();t.reset(torch.tensor([0]),torch.zeros(1))
    assert t.riser_work_proxy_J[0]==0 and t.riser_work_proxy_J[1]==before
    assert not t.previous_velocity_valid[0] and t.previous_velocity_valid[1]


@pytest.mark.parametrize('kind',['loaded_braking','far_from_edge','open_below_board','backward'])
def test_benign_horizontal_forces_do_not_become_riser_work(kind):
    t=TransitionTracker(1,'cpu',TransitionSettings());d=sample(x=.94,speed=1.,force=20.)
    d['forces'][...,0]=-500.;d['position'][...,2]=.20
    closed=torch.ones(1,dtype=torch.bool)
    if kind=='loaded_braking':d['forces'][...,2]=1000.
    elif kind=='far_from_edge':d['position'][...,0]=.4
    elif kind=='open_below_board':
        d['position'][...,0]=1.14;d['position'][...,2]=.09;closed[:]=False
    else:d['velocity'][...,0]=-1.
    for _ in range(2):t.update(closed_riser=closed,**d)
    assert t.cost['riser_work'].sum()==0


def test_descending_lip_risk_catches_hind_wheel_but_not_normal_landing():
    s=TransitionSettings()
    def evaluate(x,vx,vz,bottom):
        t=TransitionTracker(1,'cpu',s);d=sample(x=x,force=0.,speed=vx)
        d['position'][...,2]=bottom+.09;d['velocity'][...,2]=vz
        t.update(**d);return t
    assert evaluate(.94,1.,-.2,.04).cost['riser_descent']>0
    assert evaluate(.94,1.,-.2,.13).cost['riser_descent']>0
    assert evaluate(.94,1.,-.2,.15).cost['riser_descent']>0
    assert evaluate(.94,1.,-.8,.18).cost['riser_descent']>0
    assert evaluate(.94,1.,-1.,.20).cost['riser_descent']>0
    assert evaluate(.94,1.,-1.1,.25).cost['riser_descent']>0
    assert evaluate(.94,1.,-.2,.30).cost['riser_descent']==0
    assert evaluate(.4,.2,-.2,.03).cost['riser_descent']==0
    assert evaluate(.84,.2,-.2,.03).cost['riser_descent']==0 # Lands before envelope reaches next lip.
    assert evaluate(.94,1.,+.2,.04).cost['riser_descent']==0
    assert evaluate(2.1,1.,-.2,1.0).cost['riser_descent']==0 # Passed final riser.


def test_open_board_exclusion_is_not_merely_distance_exclusion():
    d=sample(x=1.14,speed=1.,force=20.)
    d['forces'][...,0]=-500.
    # Second front is x=1.2, top=.30, underside=.265. The wheel
    # envelope reaches the front but stays wholly below the open board.
    results=[]
    for closed in (False,True):
        t=TransitionTracker(1,'cpu',TransitionSettings())
        for _ in range(2):t.update(closed_riser=torch.tensor([closed]),**d)
        results.append(t.cost['riser_work'].item())
    assert results[0]==0 and results[1]>0


def test_intentional_supported_wait_exempts_progress_clock_but_not_edge_or_support():
    t=TransitionTracker(1,'cpu',TransitionSettings());d=sample(x=0.,force=100.)
    t.stall_age[:]=3.9
    for _ in range(40):t.update(command_paused=torch.ones(1,dtype=torch.bool),**d)
    assert t.stall_age==pytest.approx(torch.tensor([3.9])) and not t.failed().item()
    d['position'][...,1]=.45;t.update(command_paused=torch.ones(1,dtype=torch.bool),**d)
    assert t.edge_failed.item()


def test_new_geometry_table_includes_deployment_case_without_mutating_width():
    from task_support.stair_quiet_v1_width.settings import cases as old_cases
    assert len(cases())==35 and len(old_cases())==30
    g=cases()[-1][0]
    assert g.closed_riser and (g.height,g.depth)==(.17,.30)


def test_strict_resume_complete_state_and_old_task_rejection(make_runner,tmp_path):
    r=make_runner();r.initialize_task({})
    r.alg.guard_updates=12;r.current_learning_iteration=11
    with torch.no_grad():r.alg.actor.safety_mlp[-1].bias.fill_(.03)
    p=tmp_path/'transition.pt';r.save(str(p))
    loaded=make_runner();loaded.load(str(p))
    assert loaded.alg.guard_updates==12 and loaded._restored
    assert all(torch.equal(v,loaded.alg.actor.state_dict()[k]) for k,v in r.alg.actor.state_dict().items())
    for group in ('case_metrics','mechanics','transition_statistics'):
        assert set(course_state(loaded.env)[group])==set(course_state(r.env)[group])
    with pytest.raises(ValueError,match='contract differs'):make_runner().load(str(ROOT/runtime.DEFAULT_START))
    saved=torch.load(p,weights_only=True);del saved['m20_training_manifest']['transition_statistics']
    bad=tmp_path/'missing.pt';torch.save(saved,bad)
    with pytest.raises(ValueError,match='Incomplete Transition'):make_runner().load(str(bad))


def test_deployment_loader_keeps_scan_version_and_flat_behavior(make_runner,tmp_path,monkeypatch):
    sys.path.insert(0,str(ROOT/'deploy/deploy_mujoco'))
    import deploy_quiet_landing as deploy
    from test_safe_mujoco_runtime import scene
    from safe_actor_adapter import load_safe_actor
    from transition_actor_adapter import TransitionMujocoAdapter
    r=make_runner();r.initialize_task({})
    saved={'actor_state_dict':r.alg.actor.state_dict(),'m20_training_manifest':r._record_manifest()}
    with pytest.raises(ValueError):load_safe_actor(saved['actor_state_dict'],saved['m20_training_manifest'])
    args,actor,model,c,data,execution=scene(tmp_path,monkeypatch,saved,terrain='flat')
    assert isinstance(execution.safe,TransitionMujocoAdapter)
    obs=c.observe(data,torch.zeros(3).numpy(),torch.zeros(16).numpy())
    obs[57:]+=torch.linspace(-.1,.1,187).numpy()
    inputs=execution.policy_inputs(obs,torch.zeros(3).numpy(),actor,data)
    assert torch.equal(inputs['locomotion_scan'],inputs['policy'][:,57:])
    assert torch.equal(actor(inputs),MLPModel.forward(actor,{'policy':inputs['policy']}))
    invalid=deepcopy(saved);invalid['m20_training_manifest']['input_routing_version']='old'
    torch.save(invalid,args.checkpoint)
    with pytest.raises(ValueError,match='scan'):deploy.load_policy(args.checkpoint)


def test_last_case_accounting_and_protocol_geometry_totals(make_runner):
    from task_support.stair_quiet_v1_transition.course import record_episodes
    r=make_runner();env=r.env;state=course_state(env)
    env.scene.terrain.terrain_types[:]=34;env.scene.terrain.terrain_levels[:]=9
    env.episode_length_buf=torch.ones(4,dtype=torch.long)
    env.tail_tracker=TransitionTracker(4,'cpu',env.cfg.stair_tail)
    env.tail_tracker.success[:]=True
    env.termination_manager=NS(terminated=torch.ones(4,dtype=torch.bool),time_outs=torch.zeros(4,dtype=torch.bool),
        active_terms=['tail_success'],get_term_cfg=lambda name:NS(time_out=False))
    state['protocol'].mode[:]=torch.arange(4)
    result=record_episodes(env,torch.arange(4))
    assert state['width_statistics']['episodes'][9,34]==4 and state['episodes'][34]==4
    assert state['transition_statistics']['episodes'][:,34].tolist()==[1.,1.,1.,1.]
    assert result['Transition/pause/deployment_17_30_closed/successes_total']==1
    validate_transition_counts(state)
    state['transition_statistics']['episodes'][0,34]=.5
    with pytest.raises(ValueError):validate_transition_counts(state)
