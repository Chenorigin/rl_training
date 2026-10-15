"""CPU observers/config checks; no integrated physics or model optimization."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import math
import mujoco

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('noise_ablation',ROOT/'scripts/tools/analyze_student_descent_noise.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


class ObserverTests(unittest.TestCase):
    def test_v0_timing_is_isolated_from_noise_level(self):
        p=mod.controls({'perception':{'level':.25}})
        def timing(v):
            c=v['perception'];return tuple(math.ceil(c[k]*c['level']/.02) for k in ('map_update_period_s','max_delay_s'))
        self.assertEqual(timing(p['student-v1']),(2,1))
        self.assertEqual(timing(p['student-v0']),(5,3))
        self.assertEqual(timing(p['only_map_timing']),(5,3))
        self.assertEqual(timing(p['only_map_error_quality']),(2,1))
        for k,v in p['student-v1']['perception'].items():
            if k not in ('map_update_period_s','max_delay_s'):
                self.assertEqual(p['only_map_timing']['perception'][k],v)
        for name in ('only_action_delay','only_visibility','only_proprio'):
            self.assertEqual(p[name]['perception'],p['student-v1']['perception'])

    def test_distance_and_contact_observers_detect_both_signs(self):
        xml='<mujoco><worldbody>'+''.join(f'<body name="{name}" pos="{x} {y} 0"><geom type="sphere" size=".1"/><geom type="box" size=".5 .5 .5" contype="0" conaffinity="0"/></body>' for name,x,y in [('fl_leg',.3,.3),('fr_leg',.3,-.3),('hl_leg',-.3,.3),('hr_leg',-.3,-.3)])+'</worldbody></mujoco>'
        model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        observer=mod.LimbDistance(model)
        self.assertEqual([len(v) for v in observer.groups.values()],[1,1,2])
        self.assertGreater(observer.gaps(data)['left'],0)
        a,b=observer.groups['left'][0]
        data.geom_xpos[b]=data.geom_xpos[a]+[.15,0,0]
        self.assertAlmostEqual(observer.gaps(data)['left'],-.05)
        contacts=[SimpleNamespace(geom1=int(a),geom2=int(b),dist=-.05,efc_address=0),SimpleNamespace(geom1=int(a),geom2=int(b),dist=.05,efc_address=0),SimpleNamespace(geom1=int(a),geom2=int(b),dist=-.05,efc_address=-1)]
        with patch.object(mujoco,'mj_contactForce') as force:
            force.side_effect=lambda model,data,index,out: out.__setitem__(0,5.)
            found,peak=observer.contacts(SimpleNamespace(contact=contacts))
        self.assertEqual(found,{'left':True,'right':False,'cross':False})
        self.assertEqual(peak,5.)
        force.assert_called_once()


if __name__=='__main__':unittest.main()
