"""Fail-closed execution metadata for nominal policies using landing control.

The MLP still emits 16 nominal actions. It is not a complete robot controller.
These guards protect supported loaders/exporters, not manually stripped state
dictionaries. Real SDK deployment is deliberately not enabled by this module.
"""
from __future__ import annotations

from copy import deepcopy

from .landing import LandingSettings, control_contract, contract_hash

STUDENT_TASK = "Rough-Deeprobotics-M20-StairQuiet-v1-Student"
RUNTIME_SCOPE = ("previously_observed_noisy_terrain_memory_with_original_capture_age; "
                 "simulation_oracle_pose_and_contact; hardware_controller_not_validated")


def _markers(document):
    if not isinstance(document, dict):
        return []
    markers = []
    if "quiet_execution_control" in document:
        markers.append(document["quiet_execution_control"])
    noise = document.get("noise_refinement")
    if isinstance(noise, dict) and "control_contract" in noise:
        markers.append(noise["control_contract"])
    for key in ("m20_training_manifest", "teacher_manifest", "student_v1",
                "provenance", "training_provenance"):
        value = document.get(key)
        if isinstance(value, dict):
            markers.extend(_markers(value))
    return markers


def has_execution_control(document):
    return bool(_markers(document))


def validate_control_contract(contract):
    """Require the exact settings, semantics and source of the local core."""
    if not isinstance(contract, dict) or not isinstance(contract.get("settings"), dict):
        raise ValueError("Missing or malformed quiet execution control contract")
    try:
        expected = control_contract(LandingSettings(**contract["settings"]))
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid quiet landing settings") from error
    if contract != expected or contract_hash(contract) != contract_hash(expected):
        raise ValueError("Quiet execution control source/settings/semantics changed")
    return deepcopy(contract)


def execution_contract(document, *, required=False):
    markers = _markers(document)
    if not markers:
        if required:
            raise ValueError("Quiet checkpoint is missing its execution control contract")
        return None
    contract = validate_control_contract(markers[0])
    if any(validate_control_contract(marker) != contract for marker in markers[1:]):
        raise ValueError("Conflicting teacher/student quiet execution contracts")
    return contract


def reject_unsupported_quiet_checkpoint(document, consumer):
    # Presence, not truthiness: a null/empty marker is still not legacy.
    if has_execution_control(document):
        raise ValueError(f"{consumer} cannot execute quiet landing control; use the dedicated StairQuiet runtime")


def require_runtime(document, runtime_contract, *, platform="simulation"):
    saved = execution_contract(document)
    if saved is None:
        return None
    if platform != "simulation":
        raise ValueError("Quiet hardware execution is not approved: contact/actuator inputs remain unvalidated")
    current = validate_control_contract(runtime_contract)
    if current != saved:
        raise ValueError("Runtime would change or omit the checkpoint's quiet execution control")
    provenance = document.get("student_v1", {}).get("provenance")
    if provenance is None:
        provenance = document.get("training_provenance")
    if isinstance(provenance, dict) and "quiet_execution_control" in provenance:
        validate_provenance(provenance)
    return saved


def add_execution_provenance(provenance, contract):
    result = deepcopy(provenance)
    result["quiet_execution_control"] = validate_control_contract(contract)
    result["quiet_execution_control_sha256"] = contract_hash(contract)
    result["nominal_action_label"] = "complete teacher nominal 16D before shared execution controller"
    result["quiet_runtime_scope"] = RUNTIME_SCOPE
    return result


def validate_provenance(provenance):
    if provenance.get("task") == "Rough-Deeprobotics-M20-Student-v3-Safe":
        from task_support.student_v3_safe.deployment import validate_provenance as validate_safe
        return validate_safe(provenance)
    control = execution_contract(provenance, required=True)
    if provenance.get("quiet_execution_control_sha256") != contract_hash(control):
        raise ValueError("Quiet student execution contract digest mismatch")
    if provenance.get("task") != STUDENT_TASK:
        raise ValueError("Quiet student task/contract disagree")
    return control
