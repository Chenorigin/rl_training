"""Read-only MuJoCo gait measurements. Never changes observations or controls."""
from __future__ import annotations

import mujoco
import numpy as np

LEG_NAMES = ("fl", "fr", "hl", "hr")


class GaitMetrics:
    def __init__(self, contract, dt=0.02, min_extension=0.30):
        self.contract, self.model, self.dt = contract, contract.model, dt
        self.hips = np.array([self.model.body(f"{leg}_hipy").id for leg in LEG_NAMES])
        self.knees = np.array([self.model.joint(f"{leg}_knee_joint").qposadr[0] for leg in LEG_NAMES])
        self.min_extension = min_extension
        self.samples = []
        self.landings = []
        self.contact_time = np.zeros(4)
        self.credited_ground = np.full(4, np.nan)
        self.airborne = np.zeros(4, dtype=bool)
        self.last_leg = [-1, -1]
        self.last_level = [0, 0]
        self.same_tread_events = [0, 0]
        self.same_tread_paid = [set(), set()]
        self.alternation_checks = [0, 0]
        self.alternation_correct = [0, 0]
        self.impulse = np.zeros(4)
        self.energy = 0.0

    def forces(self, data):
        force = np.zeros((4, 3))
        wheels = self.contract.wheel_body_ids
        for i in range(data.ncon):
            contact = data.contact[i]
            body1, body2 = self.model.geom_bodyid[[contact.geom1, contact.geom2]]
            if body1 != 0 and body2 != 0:
                continue
            body = body2 if body1 == 0 else body1
            matching = np.flatnonzero(wheels == body)
            if not len(matching):
                continue
            local = np.zeros(6)
            mujoco.mj_contactForce(self.model, data, i, local)
            world = contact.frame.reshape(3, 3).T @ local[:3]
            # Contact force points from geom1 to geom2; select force on wheel.
            force[matching[0]] += world if body2 == body else -world
        return force

    def update(self, data, command):
        wheels = data.xpos[self.contract.wheel_body_ids]
        ground = np.array([self.contract.ground_height(data, *p[:2]) for p in wheels])
        forces = self.forces(data)
        clearance = wheels[:, 2] - ground - 0.09
        supporting = ((forces[:, 2] > 5)
                      & (forces[:, 2] > 0.5 * np.linalg.norm(forces, axis=1))
                      & np.isfinite(ground) & (np.abs(clearance) <= 0.05))
        self.contact_time = np.where(supporting, self.contact_time + self.dt, 0)
        stable = self.contact_time >= 0.06
        self.airborne |= ~supporting
        extension = np.linalg.norm(wheels - data.xpos[self.hips], axis=1)
        reach = np.linalg.norm((wheels - data.xpos[self.hips])[:, :2], axis=1)
        knee = data.qpos[self.knees].copy()
        yaw_only = abs(command[2]) > 0.1 and np.linalg.norm(command[:2]) < 0.1
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_BODY,
                                self.contract.base_id, velocity, 1)
        rear_valid = stable[2:] & (self.contact_time[2:] >= 0.12)
        self.samples.append(dict(time=float(data.time), command=np.asarray(command).tolist(),
            extension=extension.tolist(), knee=knee.tolist(), reach=reach.tolist(),
            clearance=clearance.tolist(), support=stable.tolist(), rear_valid=rear_valid.tolist(),
            yaw_only=bool(yaw_only), yaw_rate=float(velocity[2]),
            velocity_error=float(np.linalg.norm(velocity[3:5] - command[:2])),
            wheel_ground=ground.tolist(),
            wheel_xyz=wheels.copy().tolist(), forces=forces.tolist()))
        self.impulse += np.maximum(forces[:, 2], 0) * self.dt
        torques = data.actuator_force[self.contract.actuator_ids]
        speeds = data.qvel[self.contract.qvel_ids]
        self.energy += float(np.sum(np.abs(torques * speeds))) * self.dt
        kind, p = self.contract.terrain.kind, self.contract.terrain.parameters
        if kind == "ascent":
            h, count = p["step_height"][0], int(p["step_count"][0])
            levels = np.rint(ground / h).astype(int)
            # This instrument uses known XML step geometry, independent of the training ledger.
            for leg in range(4):
                changed = not np.isfinite(self.credited_ground[leg]) or abs(ground[leg] - self.credited_ground[leg]) > h / 2
                if stable[leg] and (changed or self.airborne[leg]):
                    self.landings.append(dict(time=float(data.time), leg=LEG_NAMES[leg],
                        level=int(levels[leg]), xyz=wheels[leg].tolist(), supporting_force=float(forces[leg, 2])))
                    axle, side = leg // 2, leg % 2
                    if levels[leg] > self.last_level[axle]:
                        if self.last_leg[axle] >= 0:
                            self.alternation_checks[axle] += 1
                            self.alternation_correct[axle] += int(side != self.last_leg[axle] and levels[leg] == self.last_level[axle] + 1)
                        self.last_leg[axle], self.last_level[axle] = side, int(levels[leg])
                    self.credited_ground[leg] = ground[leg]
                    self.airborne[leg] = False
            for axle in range(2):
                pair = slice(2 * axle, 2 * axle + 2)
                level = int(levels[2 * axle])
                # Exclude the top landing/platform: dual support there is normal.
                if stable[pair].all() and levels[pair][0] == levels[pair][1] and 0 < level < count:
                    if level not in self.same_tread_paid[axle]:
                        self.same_tread_events[axle] += 1
                        self.same_tread_paid[axle].add(level)

    def summary(self):
        if not self.samples:
            return {"sample_count": 0, "metrics_valid": False}
        ext = np.array([s["extension"] for s in self.samples])
        knee = np.abs(np.array([s["knee"] for s in self.samples]))
        clear = np.array([s["clearance"] for s in self.samples])
        reach = np.array([s["reach"] for s in self.samples])
        turn = np.array([s["yaw_only"] for s in self.samples])
        support = np.array([s["rear_valid"] for s in self.samples])
        duration = len(self.samples) * self.dt
        yaw = np.array([s['yaw_rate'] for s in self.samples])
        commanded_yaw = np.array([s['command'][2] for s in self.samples])
        return dict(sample_count=len(self.samples), metrics_valid=True,
            scope="+X ascent alternation uses XML step labels; top platform excluded; no route-bypass proof",
            rear_extension_min_m=float(ext[:, 2:].min()),
            rear_supported_extension_p05_m=float(np.quantile(ext[:, 2:][support], 0.05)) if support.any() else None,
            rear_supported_knee_abs_p95_rad=float(np.quantile(knee[:, 2:][support], 0.95)) if support.any() else None,
            rear_compressed_support_fraction=float(((ext[:, 2:] < self.min_extension) & support).sum() / support.sum()) if support.any() else None,
            compression_threshold_m=self.min_extension,
            turn_clearance_p95_m=float(np.quantile(clear[turn], 0.95)) if turn.any() else None,
            turn_reach_p95_m=float(np.quantile(reach[turn], 0.95)) if turn.any() else None,
            turn_steps=int(turn.sum()), front_same_tread_events=self.same_tread_events[0],
            turn_yaw_error_mean_rad_s=float(np.abs(yaw[turn]-commanded_yaw[turn]).mean()) if turn.any() else None,
            translation_velocity_error_mean_m_s=float(np.mean([s['velocity_error'] for s in self.samples])),
            rear_same_tread_events=self.same_tread_events[1],
            front_alternation_rate=self.alternation_correct[0] / self.alternation_checks[0] if self.alternation_checks[0] else None,
            rear_alternation_rate=self.alternation_correct[1] / self.alternation_checks[1] if self.alternation_checks[1] else None,
            alternation_checks=self.alternation_checks,
            mean_absolute_mechanical_power_w=self.energy / duration,
            absolute_mechanical_energy_j=self.energy, vertical_support_impulse_ns=self.impulse.tolist())
