from pathlib import Path
from types import SimpleNamespace
import sys,json,subprocess
import numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path('deploy/deploy_mujoco').resolve()))
import deploy_mujoco as dm
import mujoco
B=Path(__file__).resolve().parent;c=json.loads((B/'criteria.json').read_text())
model=dm.load_model(Path(c['model']),SimpleNamespace(terrain=None,terrain_xml=Path(c['flat_terrain'])))
r=mujoco.Renderer(model,height=360,width=480);datasets=[]
for label,title in [('baseline','Original 257100'),('distilled_round5','Idle-stop fitted')]:
 f=np.load(B/'evaluations'/label/'平地启停_full.npz');data=mujoco.MjData(model);base=model.body('base_link').id;i=np.searchsorted(f['time'],14.02);data.qpos[:]=f['qpos'][i];mujoco.mj_forward(model,data)
 cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE;cam.distance=3.2;cam.azimuth=170;cam.elevation=-25;cam.lookat[:]=data.xpos[base]
 datasets.append((f,data,cam,title,f['qpos'][i,:2].copy()))
out=B/'stop_before_after_v2.mp4';p=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','960x390','-r','10','-i','-','-an','-c:v','libx264','-threads','1','-pix_fmt','yuv420p',str(out)],stdin=subprocess.PIPE)
for time in np.arange(14.02,36,.1):
 canvas=Image.new('RGB',(960,390),'white');draw=ImageDraw.Draw(canvas)
 for col,(f,data,cam,title,start) in enumerate(datasets):
  i=np.argmin(abs(f['time']-time));data.qpos[:]=f['qpos'][i];data.qvel[:]=f['qvel'][i];mujoco.mj_forward(model,data);r.update_scene(data,camera=cam);r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False;r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=False;canvas.paste(Image.fromarray(r.render()),(480*col,30))
  drift=np.linalg.norm(f['qpos'][i,:2]-start)*100;draw.text((480*col+8,8),f'{title} | t={f["time"][i]:.2f}s | displacement={drift:.1f}cm',fill='black')
 p.stdin.write(np.asarray(canvas).tobytes())
 if abs(time-35.02)<.01:canvas.save(B/'stop_before_after_frame.jpg')
p.stdin.close();assert p.wait()==0;r.close();print(out)
