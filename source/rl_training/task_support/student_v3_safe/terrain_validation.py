"""Bounded checks against the imported PhysX collision scene, validation only."""
import numpy as np
from .terrain_geometry import geometries, surfaces


def collision_receipt(env):
    import omni.physx
    query = omni.physx.get_physx_scene_query_interface()
    terrain = env.scene.terrain
    table = geometries(env.cfg.student_terrain_course)
    probes = []
    for row in (0, len(env.cfg.stair_tail.widths_m)-1):
        for gi in (len(table)-2, len(table)-1):
            geometry = table[gi]
            # First material, exact generated physical tile origin.
            origin = terrain.terrain_origins[row, gi*5].detach().cpu().numpy()
            spans, goal, goal_height = surfaces(geometry, env.cfg.stair_tail)
            for lo, hi, z in spans:
                start = origin+np.array([(lo+hi)/2, 0., 6.])
                hit = query.raycast_closest(tuple(start.tolist()), (0.,0.,-1.), 7.)
                if not hit['hit'] or abs(hit['position'][2]-origin[2]-z) > 2.e-4:
                    raise RuntimeError(f'Physical tread query disagrees: {geometry.name} row {row} at {start}: {hit}')
                probes.append(dict(family=geometry.family,row=row,x=float((lo+hi)/2),
                    expected_height=float(z),actual_height=float(hit['position'][2]-origin[2]),
                    collision=str(hit.get('collision','')),normal=list(hit['normal'])))
            x = env.cfg.stair_tail.start_length/2
            start = origin+np.array([x-.1, 0., geometry.height/2])
            hit = query.raycast_closest(tuple(start.tolist()), (1.,0.,0.), .2)
            expected = geometry.family != 'open_ascent'
            if bool(hit['hit']) != expected:
                raise RuntimeError(f'Physical riser query disagrees: {geometry.name}: {hit}')
            probes.append(dict(family=geometry.family,row=row,probe='first_riser',
                               expected_hit=expected,actual_hit=bool(hit['hit'])))
    return dict(probes=probes,contact_materials=terrain.contact_receipt,
                rows=len(env.cfg.stair_tail.widths_m),columns=len(table)*5,
                collision_engine='PhysX scene query on cooked static meshes')
