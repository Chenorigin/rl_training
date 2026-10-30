"""Fail CTS playback at the first SDK-equivalent position violation."""
from copy import deepcopy
import json

import numpy as np

from .joint_limits import validate_limits


class PositionLimitError(RuntimeError):
    def __init__(self, record):
        self.record = record
        super().__init__('CTS_JOINT_LIMIT_FAILURE ' + json.dumps(record, allow_nan=False))


class MujocoJointLimits:
    def __init__(self, cfg, model, contract, constants):
        import mujoco
        self.cfg = deepcopy(validate_limits(cfg))
        self.contract = contract
        self.target = np.asarray(cfg['target_bounds_rad'], dtype=np.float32)
        self.feedback = np.asarray(cfg['feedback_bounds_rad'], dtype=np.float32)
        self.default = np.asarray(cfg['default_position_rad'], dtype=np.float32)
        self.scale = np.asarray(cfg['action_scale'], dtype=np.float32)
        if not np.allclose(constants.DEFAULT_JOINT_POS[:12], self.default, atol=1.e-6, rtol=0) or not np.allclose(constants.ACTION_SCALE[:12], self.scale, atol=1.e-6, rtol=0):
            raise ValueError('MuJoCo action scale/default differs from CTS software-limit contract')
        for i, name in enumerate(cfg['joint_names']):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0 or model.jnt_qposadr[jid] != contract.qpos_ids[i]:
                raise ValueError('MuJoCo software-limit joint mapping differs: ' + name)
            if not model.jnt_limited[jid] or not np.allclose(model.jnt_range[jid], self.feedback[i], atol=1.e-6, rtol=0):
                raise ValueError('MuJoCo XML range differs from CTS software feedback limit: ' + name)
        self.last_action = None

    def _check(self, data, values, bounds, kind):
        values = np.asarray(values, dtype=np.float32).reshape(12)
        bad = ~np.isfinite(values) | (values < bounds[:, 0]) | (values > bounds[:, 1])
        if not bad.any():
            return
        q = data.qpos[self.contract.qpos_ids[:12]]
        def finite(value):
            return float(value) if np.isfinite(value) else None
        raise PositionLimitError({
            'status': 'FAIL', 'reason': kind, 'time_s': float(data.time),
            'action': None if self.last_action is None else [finite(x) for x in self.last_action],
            'violations': [{'joint': self.cfg['joint_names'][i], 'value_rad': finite(values[i]),
                            'measured_rad': finite(q[i]), 'lower_rad': float(bounds[i, 0]),
                            'upper_rad': float(bounds[i, 1])} for i in np.flatnonzero(bad)],
        })

    def check_action(self, data, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (16,) or not np.isfinite(action).all():
            raise ValueError('CTS requires finite 16D actions')
        self.last_action = action.copy()
        self.check_feedback(data)
        target = self.default + np.clip(action[:12], *self.cfg['normalized_action_clip'])*self.scale
        self._check(data, target, self.target, 'nominal_target')

    def check_feedback(self, data):
        self._check(data, data.qpos[self.contract.qpos_ids[:12]], self.feedback, 'measured_position')

    def check_corrected(self, data, target):
        self._check(data, target, self.target, 'corrected_target')

    def metadata(self):
        return {'software_joint_limits': {'enabled': True, 'contract': self.cfg,
                'checks': ['issued_target', 'applied_target', 'corrected_target', 'substep_feedback'],
                'target_clipping': False}}
