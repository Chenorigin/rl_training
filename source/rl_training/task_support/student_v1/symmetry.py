"""Sagittal reflection for the pinned M20 leg/action and map ABI."""
import torch

# Leg blocks FL, FR, HL, HR. Reflection negates abduction only;
# wheel axes and flexion axes have identical left/right joint conventions.
PERM = (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8, 13, 12, 15, 14)
SIGN = (-1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, 1, 1)


def mirror_actions(value):
    if value.shape[-1] != 16:
        raise ValueError("Expected M20 16D action/joint vector")
    return value[..., list(PERM)] * value.new_tensor(SIGN)


def mirror_proprio(value):
    if value.shape[-1] != 57:
        raise ValueError("Expected M20 57D proprioception")
    result = value.clone()
    # Angular velocity is axial, gravity and linear commands are polar.
    result[..., :3] = value[..., :3] * value.new_tensor([-1, 1, -1])
    result[..., 3:6] = value[..., 3:6] * value.new_tensor([1, -1, 1])
    result[..., 6:9] = value[..., 6:9] * value.new_tensor([1, -1, -1])
    for start in (9, 25, 41):
        result[..., start:start+16] = mirror_actions(value[..., start:start+16])
    return result


def mirror_observations(value):
    dim = value.shape[-1]
    if dim != 244 and (dim < 618 or (dim-618) % 57):
        raise ValueError("Expected teacher 244D or student 618+57*k ABI")
    result = value.clone()
    result[..., :57] = mirror_proprio(value[..., :57])
    for start in ((57,) if dim == 244 else (57, 244, 431)):
        result[..., start:start+187] = value[..., start:start+187].reshape(*value.shape[:-1], 11, 17).flip(-2).flatten(-2)
    if dim > 618:
        result[..., 618:] = mirror_proprio(value[..., 618:].reshape(*value.shape[:-1], -1, 57)).flatten(-2)
    return result
