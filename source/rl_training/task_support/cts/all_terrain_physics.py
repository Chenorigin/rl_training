"""Per-environment continuous physical draws; no shared material mutation."""
import torch


class ContinuousPhysics:
    def __init__(self, env, cfg):
        self.env, self.cfg = env, cfg
        self.robot = env.scene['robot']
        view = self.robot.root_physx_view
        self.mass = view.get_masses().clone()
        self.inertia = view.get_inertias().clone()
        self.com = view.get_coms().clone()
        self.base_body = self.robot.find_bodies('base_link')[0]
        if len(self.base_body) != 1:
            raise ValueError('Continuous COM randomization requires exactly one base_link')
        self.gains = [(a, a.stiffness.clone(), a.damping.clone()) for a in self.robot.actuators.values()]
        self.values = torch.zeros(env.num_envs, 7, device=env.device)
        self.readbacks = 0

    def apply(self, ids, levels):
        if not len(ids):
            return
        cfg, view = self.cfg, self.robot.root_physx_view
        cpu = ids.to(device='cpu', dtype=torch.long)
        scale = cfg['initial_scale']+(1-cfg['initial_scale'])*levels.float()/2
        values = []
        for name in ('static_friction', 'dynamic_ratio', 'mass_scale', 'kp_scale', 'kd_scale'):
            lo, hi = cfg[name]
            centre = 1. if name.endswith('scale') else (lo+hi)/2
            lower, upper = centre+(lo-centre)*scale, centre+(hi-centre)*scale
            values.append(lower+torch.rand(len(ids), device=ids.device)*(upper-lower))
        friction, ratio, mass, kp, kd = values
        material = view.get_material_properties().clone()
        material[cpu, :, 0] = friction.cpu()[:, None]
        material[cpu, :, 1] = (friction*ratio).cpu()[:, None]
        material[cpu, :, 2] = 0.
        # Direct physical tensor assignment, not random selection of buckets.
        view.set_material_properties(material, cpu)
        masses = view.get_masses().clone()
        inertias = view.get_inertias().clone()
        masses[cpu] = self.mass[cpu]*mass.cpu()[:, None]
        inertias[cpu] = self.inertia[cpu]*mass.cpu()[:, None, None]
        view.set_masses(masses, cpu)
        view.set_inertias(inertias, cpu)
        com = view.get_coms().clone()
        offsets = (torch.rand(len(ids), 3, device=ids.device)*2-1)*cfg['com_offset_m']*scale[:, None]
        com[cpu] = self.com[cpu]
        com[cpu[:, None], self.base_body, :3] += offsets.cpu()[:, None]
        view.set_coms(com, cpu)
        for actuator, base_kp, base_kd in self.gains:
            actuator.stiffness[ids] = base_kp[ids]*kp[:, None]
            actuator.damping[ids] = base_kd[ids]*kd[:, None]
        actual = view.get_material_properties()[cpu, :, :2]
        if not torch.allclose(actual, material[cpu, :, :2], atol=1e-6, rtol=1e-5):
            raise RuntimeError('Continuous friction did not reach PhysX')
        if not torch.allclose(view.get_masses()[cpu], masses[cpu], atol=1e-5, rtol=1e-5):
            raise RuntimeError('Continuous mass did not reach PhysX')
        if not torch.allclose(view.get_inertias()[cpu], inertias[cpu],atol=1e-6,rtol=1e-5):
            raise RuntimeError('Continuous inertia did not reach PhysX')
        if not torch.allclose(view.get_coms()[cpu],com[cpu],atol=1e-6,rtol=1e-5):
            raise RuntimeError('Continuous COM did not reach PhysX')
        self.values[ids] = torch.stack((friction, friction*ratio, mass, kp, kd, offsets[:, 0], offsets[:, 1]), -1)
        self.readbacks += len(ids)

    def friction(self):
        return self.values[:, :2]

    def diagnostics(self):
        names = ('static_friction', 'dynamic_friction', 'mass_scale', 'kp_scale', 'kd_scale', 'com_x', 'com_y')
        result = {'physical_readback_environments': self.readbacks}
        for index, name in enumerate(names):
            result[name+'_min'] = float(self.values[:, index].min())
            result[name+'_max'] = float(self.values[:, index].max())
        return result
