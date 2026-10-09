# cyq rl_training: gait196600 and realistic outdoor terrain (2026-10-07)

Scope: /home/ubuntu/cyq_shixi_projects/rl_training only; this is not PGTT from other projects. No teacher rewards, pre_teacher, training configuration or deploy controls changed. Frozen target model196600 from2026-10-06_22-06-56_stair_resume, compare80000 and149999. Full evidence: docs/gait_quality_20261007/README_CN.md and analysis.json.

## 现象
User sees front folding on ascent, rear forward lean at descent entry, wants strict adjacent-riser alternating front/rear landing, contact/energy assessment and more realistic XML.

## 真因 / 已证事实
Original FSM headless CPU physical rollouts3models×3conditions,8risers15cm,vx0.5. Latest clears short wide/narrow ascent and descent. Both axles wide tread repeatedly FL1FR1FL2FR2, HL1HR1HL2HR2. Low simultaneous same-tread cost is not evidence of strict temporal alternation. Ascent target retires after first contact, pair cost needs simultaneous contact; sequential opposite-leg following can avoid pair definition.
Front wide support not over canonical envelope, narrow swing maximum157deg and minextension0.185m. Descent rear forward reach0.318m, support knee max83deg and extensionmin0.409m: foreaft posture is different from fold length/knee cost.
Latest versus80000 wide ascent energyperdistance down9.4%, forcep95 down15.4% but max up5.8%; narrow max up46.4%, p99 up35.3% despite energyperdistance down21.6%. Latest versus149999 energyperdistance rises in both ascents and descent.80000descent leaves course, not an equivalent completion energy baseline.
Latest TB500 weightedEpisode_Reward mechanicalpower~-0.000719 vs velocitytrack4.518. Energy terms nonzero. Mean over16joints before1800normalization weakens totalpower interpretation. All baselines already energyreward; no causal ablation conclusion.

## 判据
200Hz actual wheel world contact forces, mechanical work sumabs(actualtau*jointvel), matched phase and completed task; static gravity scale checked.50Hz first stable landing perleg/riser records sequential same-tread, floor/top excluded. Mechanical proxy not battery, one run not success rate. Real open-loop GUI/keyboard path and full44m outdoor traversal not tested.

## 修法 / 交付
Added separate deploy/deploy_mujoco/terrains/realistic_outdoor.xml with curb, seams, building stair flight, mildly tilted slabs, selfcontained embeddedhfield dirt, rounded fixed stones, gentlehill, wetfriction. Existingloader used, actor244→16 and robotABI/mass unchanged.26independentraychecks incl asymmetrichfield vertices,28scanposes, wetactualcontactmu.45, 50step originalentry physical smoke and inspectedoffscreenpreview.
MuJoCo native hfield elevation source row order is flipped on compile; confirmed with asymmetric vertex values and corrected XML annotation(Y+toY- source rows).No deployment scan fix needed.
Future suggestions not implemented: retain per-riser temporal landing ledger, phase/clearance-aware front fold envelope, descent-entry foreaft reach envelope, then power-scale/impact-proxy calibration with frozen matched tests. Do not blindly tighten all knee angles or claim success from low reward curves.

KB synchronization: shared knowledge repo previously has corrupt looseobject fabee48262ee49c5e328a344da94c08346f4a7d5; local knowledge write allowed, remote fetch/push blocked. Do not repair or overwrite unrelated shared changes during this task.
