# M20 stair gait safety reward revision 2026-10-07

Scope: cyq_shixi_projects/rl_training only, no PGTT. Full report docs/gait_safety_20261007/README_CN.md. Single numeric authority production stair_teacher.py TUNING.

## 现象
User confirmed a small in stairs_ascent.xml loaded/swing frames, rear downward-entry foreaft pose and temporal same-tread following. Model196600 passed20stairs but style failed.

## 真因
Oldfront .20s grace bypassed loaded .02s frame; old swing q2.5rad plus .5rate allowed deep swing; rear knee/length could not constrain foreaft extension. Old pair cost required simultaneous contact; independent CPU FL1leaves thenFR1/HL1leaves thenHR1 gotzero. Runtime robotIDs13-16 and sensorIDs4,8,12,16 are different entity orderings, not misbinding: actual name assertions both resolveFL/FR/HL/HR. Retracted provisional parent-overwrite hypothesis without production changes.

## 判据
Three actual CPU suites pass; before/after same production function body counterexamples; independent mirrored/short-contact/multiflight/strictmetrics/progress checks pass. ActualIsaac16env250step smoke then finalexactsource100step, actor244/critic285/action16 finite, realentity names asserted. Frozenpre/deploy/agents/XML hashes unchanged. Oldrecordedposes hypotheticalnewcosts separated from new-policyperformance. No optimizer updates or newweights, no claimed gaitrecovery. Samefamilyindependentreview, heterogeneous unavailable.

## 修法
Front fold monotonicsoft excess with immediate loadgain and rise-relaxed phase envelopes; separate9point front underside verticalgap proxy. Rear body/gravity forwardangle excess cost while loadeddescend.64physicaltreads eachaxle remembertwo stablelanding bits, firstretired arrival seedsimmediately; intermediacy cached, terminalplatform exempt, onceperaxle/riserduplicatepaid, currentbodyyaw cannot exemptforward-command duplicates. Wrong side completion no credit; directskip stillno completioncreditbutnoexplicitcost. Newflightonly afterfour-wheelclearplatform and distantnewtarget; oldtopnotmisclassified. strict localmetric adds ordervalid plusnonduplicate, report denominator.

Recommendedwarmstart checked196600actor using existing--init_actor_from, freshcritic/Adam/iteration (changedreturns), fixedsmallLR/clip via commanddocumented. Directorylatest199998untestednotautomaticallybetter.149999energycomparisonbaseline, not provenbetterstartingpoint. Recenttrainingvaluabletaskability; don't restartprewithoutfixedmatched evidence. Training30cmtread/test20cm distributiondifference remains. True25cm strictsingle-legtwo-riserfeasibility unverified.

Reviewpitfall: mechanicalstringreplacement accidentally insertednew_flight block into unrelatedprogress/descentfunction. Independent directprogress invocation foundNameError despite3CPUtests passing; removed, added realcalls. Never equate compile/simpletest success with allactive rewardpaths covered.
