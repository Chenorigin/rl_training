from pathlib import Path
from array import array
import hashlib,json
import numpy as np
from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader

OUT=Path(__file__).resolve().parent
run=Path('logs/rsl_rl/deeprobotics_m20_quiet_landing_v1_5/2026-10-10_17-55-27_quiet_landing_v1_5_from_baseline')
data={};manifest=[]
for source in sorted(run.glob('events.out*')):
    size=source.stat().st_size;left=size;sha=hashlib.sha256();snapshot=OUT/'事件文件冻结.tmp'
    with source.open('rb') as src,snapshot.open('wb') as dst:
        while left:
            block=src.read(min(left,1024*1024))
            if not block:break
            dst.write(block);sha.update(block);left-=len(block)
    assert left==0
    count=0
    for event in LegacyEventFileLoader(str(snapshot)).Load():
        count+=1
        for value in event.summary.value:
            if value.WhichOneof('value')=='simple_value':
                data.setdefault(value.tag,array('d')).extend([event.step,event.wall_time,value.simple_value])
    manifest.append({'file':str(source),'frozen_bytes':size,'prefix_sha256':sha.hexdigest(),'parsed_events':count})
    snapshot.unlink()
data={k:np.asarray(v).reshape(-1,3) for k,v in data.items()}
np.savez_compressed(OUT/'V1_5当前全部标量.npz',**data)
selected=['QuietLanding/all/pre_speed_mean_m_s','QuietLanding/all/specific_energy_mean_J_kg',
          'QuietLanding/specific_energy_per_m_J_kg_m','QuietLanding/impact_window_force_max_N',
          'QuietLanding/impact_window_excess_impulse_max_N_s','QuietLanding/events_per_m',
          'Metrics/base_velocity/error_vel_xy','Metrics/base_velocity/error_vel_yaw',
          'Train/mean_episode_length','Task/episode_reward_rate','Loss/value','Loss/anchor',
          'Policy/mean_std','QuietLoad/stable_support_fraction','QuietLoad/stable_diagonal_abs_mean',
          'QuietReward/quiet_load_balance','QuietReward/quiet_impact_peak']
stats={}
for step in [230300,233500,236200]:
    stats[str(step)]={}
    for key in selected:
        a=data[key];v=a[(a[:,0]>=step-1000)&(a[:,0]<step),2]
        assert len(v) and np.isfinite(v).all(),key
        stats[str(step)][key]={'samples':len(v),'mean':float(v.mean()),'median':float(np.median(v))}
result={'source':manifest,'scalar_tags':len(data),'max_log_iteration':int(max(v[-1,0] for k,v in data.items() if not k.endswith('/time'))),
        'window_definition':'Each checkpoint iteration k: [k-1000,k), equal 1000-update means. Training includes failures and curriculum; force peak is mean of reset-batch maxima.',
        'statistics':stats,'new_standing_route_tags_present':any(k.startswith(('QuietStand/','QuietRoute/')) for k in data)}
(OUT/'TensorBoard分窗口统计.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
