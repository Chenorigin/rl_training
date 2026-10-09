# Fixed comparison, frozen before simulation
Models: baseline 196600 vs gait_safety_v2 22000, immutable copies and SHA256 in checkpoints.json.
Sequential CPU, torch/BLAS one thread, no training or reward modification; <3 GB RAM/<100 MB disk/<10 min whole batch. Live machine has ~106 GB available RAM, training GPU ~6.4/49.1 GB occupied; no evaluation GPU.
Cases (one deterministic run each, NOT estimates of success rates):
1. xml20: unchanged stairs_ascent.xml, 20x15cm risers/20cm tread, W=0.7, 2200 controls limit.
2. up30: 8x15cm/30cm, W=0.5, 1600 controls.
3. down30: 8x15cm/30cm, W=0.5, 1600 controls.
4/5. turn_q/turn_e: unchanged flat.xml, Q/E=+/-0.6 rad/s, 800 controls.
Original prone/Z at0.25s/C at5.8s/movement at6.5s FSM, PD/action/actor unchanged.
Completion: all wheels beyond goal+wheel_radius within course width, no original-policy fall (root-ground<0.22m or Rzz<=0.35). Route exit fails. Timeout fails stairs, normal turn duration completes turn.
Geometry only on copied MjData refreshed by mj_kinematics; forces from original physics contacts.
Front stair-stage stats exclude floor and terminal platform, separately loaded support, >=0.2s support, force<5N swing. Knee internal hinge a=180-|q|; target a>=90 support/a>=80 swing at15cm, soft preferences. Front body gap is 9-point vertical underside terrain proxy with0.22m target, not true collision distance.
Rear descent-entry support >=0.12s, body b and gravity/yaw g thresholds35/25deg.
Landing >=0.06s stable per wheel, first credit each (leg,riser), chronological axle sequence; correct transition iff opposite leg AND next consecutive riser. Both front and rear must satisfy ALL transitions, cover intermediate risers, and avoid duplicates to claim strict alternating. Proxy may miss contacts at edges, report missing instead of silently pass.
Physical force magnitude P95/P99/peak at200Hz; absolute mechanical energy/distance only compare completed same cases. These are mechanical proxies, not battery consumption.
Turns: actual yaw velocity error, ground wheel clearance and reach, translation drift; duration alone not tracking success.
Decision: loss of any baseline task => regression; gait improvement requires actual angle/sequence improvement without crossing regression; overall reward alone insufficient. Unresolved/under-sampled fields marked unavailable.
No optional expanding the test set or changing thresholds after results.
