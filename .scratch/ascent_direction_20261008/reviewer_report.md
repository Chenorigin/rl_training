# Independent second review, 2026-10-08

Scope: read-only review of restored reward baseline plus the new ascent direction cost. Same model family; no claim of heterogeneous model review. No Isaac, optimization, or physical-policy evaluation launched by this reviewer.

Executed:

- `check_ascent_direction.py`: pass. Existing production-function checks cover mirrored drift, arbitrary world yaw, body-yaw co-rotation, commanded curved routes at lower forward speed, nonzero commanded vy, standing/reverse/pure-yaw/descent gates, temporary missing scan, once-per-step cache, reset, observed platform exit, new flight, command change, monotonic bounded finite costs and config bindings.
- `.scratch/ascent_direction_20261008/reviewer_vector.py`: pass. Eight environments over 120 randomized steps with per-environment yaw, vx/vy/wz changes, independent resets and varying visible edges. Batched cost/active/cross-track/lateral-speed/heading-error match eight separate single-environment executions exactly. Max sampled cost 0.9397614.
- `check_stair_rewards.py --assert-fixed`: fail. Restored old same-tread behavior returns same_tread_cost=False, contrary to newer assertion. Root notified; baseline attribution required. This failed check cannot be called green.
- `check_gait_refinement.py`: initially fail from missing np in AST test namespace, despite production importing numpy. Root notified; root is repairing the test fixture injection and repeating the check.

No new concrete direction-cost bug was found in this scope. Current body yaw is used to initialize a newly detected flight but does not continually redefine the integrated route. Explicit user yaw command rotates reference; sideways command defines path tangent. Along-track slowdown alone is unpenalized. Cost has a deadband and smooth saturation; three components bound total cost below 2, weighted below 4 per second before standard timestep integration.

Limits: direction gain in a learned policy is unmeasured. The end latch proves all wheels have reached the upper platform of the highest observed riser; it cannot prove this is the globally final riser if future terrain is hidden. The current term is command tracking, not guaranteed globally optimal route selection or strict terrain-edge heading alignment. Old reward vulnerabilities restored by explicit rollback should be distinguished from new regressions. Full Isaac runtime binding remains root's verification task.
