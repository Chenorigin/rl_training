"""Predictive landing shaping and immediate unsafe-load accounting."""
import torch
from task_support.stair_quiet_v1_width.signals import TailTracker
from .observations import safety_geometry


class SafeTailTracker(TailTracker):
    def __init__(self, n, device, settings):
        super().__init__(n, device, settings)
        self.cost["landing_risk"] = torch.zeros(n, device=device)
        for name in ("early_unsafe_foot_seconds", "predicted_unsafe_foot_seconds", "touchdown_unsafe_count",
                     "slip_foot_distance_m"):
            setattr(self, name, torch.zeros(n, 4, device=device))
        self.previous_supported = torch.zeros(n, 4, dtype=torch.bool, device=device)
        self.elapsed_s = torch.zeros(n, device=device)
        self.backward_distance_m = torch.zeros(n, device=device)
        self.previous_x = torch.zeros(n, device=device)

    def reset(self, ids, root_x):
        super().reset(ids, root_x)
        self.previous_x[ids] = root_x

    def update(self, **d):
        previous_unsafe_seconds = self.unsafe_seconds.clone()
        super().update(**d)
        s, dt = self.s, d["dt"]
        g = safety_geometry(d, s)
        supported = g["supported"]
        unsafe = (g["forward"] < 0) | (g["side"] < 0)
        early = supported & (self.support_age < s.landing_grace_s)
        approaching = (~supported & (d["velocity"][..., 2] < -s.prediction_descent_m_s)
            & (g["clearance"] > -.05) & (g["clearance"] < s.prediction_clearance_m))
        predicted_side = torch.minimum(g["predicted_left"], g["predicted_right"])
        risk_now = ((-g["forward"]).clamp_min(0)/.1).square()+((-g["side"]).clamp_min(0)/.1).square()
        risk_future = ((-g["predicted_forward"]).clamp_min(0)/.1).square()+((-predicted_side).clamp_min(0)/.1).square()
        self.cost["landing_risk"] += (risk_now*early+risk_future*approaching).sum(-1)*dt
        self.early_unsafe_foot_seconds += (unsafe & early)*dt
        self.predicted_unsafe_foot_seconds += (approaching & (
            (g["predicted_forward"] < 0) | (predicted_side < 0)))*dt
        self.touchdown_unsafe_count += (supported & ~self.previous_supported & unsafe)
        self.previous_supported.copy_(supported)
        # Account from first load; keep original stable-support metric separately.
        self.unsafe_foot_seconds += (unsafe & early)*dt
        self.unsafe_seconds.copy_(previous_unsafe_seconds+(unsafe & supported).any(-1)*dt)
        self.slip_foot_distance_m += g["slip_speed"]*(
            supported & (g["slip_speed"] > s.slip_speed_m_s))*dt
        self.elapsed_s += dt
        self.backward_distance_m += (self.previous_x-d["root"][:, 0]).clamp_min(0)
        self.previous_x.copy_(d["root"][:, 0])
