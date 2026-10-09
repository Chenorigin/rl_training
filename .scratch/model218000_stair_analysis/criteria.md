# model_218000 stairs analysis — frozen criteria

Checkpoint: `logs/rsl_rl/deeprobotics_m20_stair_teacher/2026-10-09_11-27-33_rear_alternation_fix/model_218000.pt`

## Fixed rollouts

- Physical closed-loop MuJoCo, official M20 XML, 200 Hz physics / 50 Hz policy.
- Constant command `(vx, vy, wz) = (0.5, 0, 0)` unless a result explicitly says otherwise.
- Ascent: 20 risers, 0.15 m rise, 0.30 m tread, at most 1000 policy steps.
- Descent: 5 risers, 0.15 m rise, 0.30 m tread, at most 750 policy steps.
- Reference checkpoint for model-change attribution: the run's actual resume source `model_214700.pt` under `2026-10-09_01-55-35_gait_fix_descent_entry`.

## Judgement

- Rear strict alternation is measured from stable first landings per physical tread. Desired examples are `HL1 -> HR2 -> HL3` or `HR1 -> HL2 -> HR3`. `HL1 -> HR1 -> HL2 -> HR2` is same-tread pairing and fails, even if the first-arrival-only metric looks alternating.
- Report front and rear chronology independently. A single deterministic rollout establishes reproduction, not a success rate.
- Descent entry is the interval after a front wheel first establishes lower-tread support and before both rear wheels settle below the top. Measure each rear wheel's hip-relative forward displacement and body/gravity forward angle, split by loaded/swing state.
- Collision risk is not inferred from angle alone. Report actual MuJoCo robot self-contact if present; otherwise report the minimum geometric separation available from collision geoms or mark it unmeasured.
- TensorBoard mixed-terrain trends diagnose learning-signal prevalence only; they do not substitute for fixed-rollout behavior.

## False positives / false negatives

- False positive for alternation: recording only the first leg to reach each new maximum height hides the partner's later same-tread landing. Use persistent per-leg/per-tread history.
- False negative for alternation: repeated contact chatter can create duplicate events. Deduplicate each leg/tread and require stable support.
- False positive for collision: wheel/hip center distance can look small while collision shapes remain separated. Require contact or collision-geom distance.
- False negative for collision: self-collision may be disabled in the training asset or omitted in the MuJoCo model. In that case, absence of contact is not proof of clearance; state the simulator limitation.

## Stop condition

Stop after checkpoint identity/config are verified, TensorBoard trends are extracted, both fixed rollouts finish, and the two reported problems have evidence-backed causal hypotheses plus concrete next measurements. No reward or training changes in this task.
