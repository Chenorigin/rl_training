import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()/'deploy/deploy_mujoco'))
import deploy_mujoco as dep
import mujoco
import numpy as np
import torch
from tensordict import TensorDict
from PIL import Image

torch.set_num_threads(1)
model_path=Path('/home/ubuntu/桌面/m20_1/M20_perception_Lidar_rl_2/M20_perception_Lidar_rl/deep_robotics_model/M20/mjcf/M20.xml')
for label,cp in [('baseline','2026-10-06_22-06-56_stair_resume/model_199998.pt'),('latest','2026-10-08_16-51-12_gait_v3/model_1700.pt')]:
 cfg=argparse.Namespace(terrain='ascent',terrain_xml=None,stair_height=.15,tread_depth=.3,stair_count=5)
 model=dep.load_model(model_path,cfg);data=mujoco.MjData(model);contract=dep.M20Contract(model,cfg.terrain_config);contract.reset(data)
 actor=dep.load_actor(Path('logs/rsl_rl/deeprobotics_m20_stair_teacher')/cp);prev=np.zeros(16,dtype=np.float32);cmd=np.array([.5,0,0],dtype=np.float32)
 for _ in range(500):
  with torch.inference_mode():action=actor(TensorDict({'policy':torch.from_numpy(contract.observe(data,cmd,prev))[None]},batch_size=[1])).squeeze().numpy()
  for _ in range(4):contract.apply_pd(data,action);mujoco.mj_step(model,data)
  prev=action.astype(np.float32)
 camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE;camera.lookat[:]=[1.3,0,.5];camera.distance=6;camera.azimuth=95;camera.elevation=-18
 renderer=mujoco.Renderer(model,height=480,width=640);renderer.update_scene(data,camera=camera)
 Image.fromarray(renderer.render()).save('docs/gait_v3_regression/'+label+'_ascent_10s.png');renderer.close()
 print(label,data.xpos[contract.base_id].tolist(),flush=True)
