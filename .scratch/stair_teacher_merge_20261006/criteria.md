Scope: merge implementations and migrate active call sites. No reward/threshold/weight/observation/action/curriculum changes. Pre teacher frozen.
Acceptance: no duplicate definitions; AST bodies equal except removed self imports; CPU reward and gait outputs exactly equal before/after; runtime smoke loads real configuration and actor checkpoint and executes finite steps; old import aliases identical to canonical objects.
Limit: equivalence and short smoke do not prove gait quality or training convergence. New full training skipped.
CPU budget: serial one-thread Torch tests, small synthetic tensors, allow 2 GiB incl imports. Files <1 MiB. GPU smoke only after live budget check.
