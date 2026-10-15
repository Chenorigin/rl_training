"""Public perception preset names, independent of the selected policy version."""

PRIMARY_PROFILE_NAMES = ("teacher", "student-v0", "student-v1")
PROFILE_ALIASES = {"clean": "teacher", "realistic": "student-v0", "train": "student-v1"}
LEGACY_PROFILE_NAMES = {value: key for key, value in PROFILE_ALIASES.items()}


def canonical_profile_name(name):
    return PROFILE_ALIASES.get(name, name)


def profile_entry(registry, name):
    """Read current or archived user profile files without changing their values."""
    canonical = canonical_profile_name(name)
    if canonical in registry:
        return registry[canonical]
    legacy = LEGACY_PROFILE_NAMES.get(canonical, canonical)
    if legacy in registry:
        return registry[legacy]
    raise ValueError(f"Missing noise profile {canonical!r} in profile file")
