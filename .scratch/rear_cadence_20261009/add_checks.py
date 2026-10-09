from pathlib import Path
p=Path('scripts/tools/check_gait_v3.py');s=p.read_text();needle='    return out\n'
add='''    # Preparation tracks raw geometry even when support eligibility is off.
    e=make_env();st=ns['_new_state'](e);row=torch.tensor([0]);target=torch.tensor([0]);has=torch.tensor([True])
    expected=torch.tensor([-1]);eligible=torch.tensor([[False,False]])
    def prep(values,elig=eligible,exp=expected):
        return float(ns['rear_preparation_gain'](st,row,target,has,torch.tensor([values]),elig,exp))
    assert prep([.2,.1])==0
    assert prep([.4,.3])==0  # loaded rolling updates maxima without paying
    assert prep([.4,.3],torch.tensor([[True,True]]))==0  # support flip only
    assert abs(prep([.6,.3],torch.tensor([[True,False]]))-.2)<1e-6
    assert prep([.6,.3],torch.tensor([[True,True]]))==0
    assert prep([.7,.8],torch.tensor([[True,True]]),torch.tensor([0]))>0
    assert prep([.7,.8],torch.tensor([[True,True]]),torch.tensor([1]))==0
    assert prep([.5,.7],torch.tensor([[True,True]]))==0
    assert prep([1.,1.],torch.tensor([[True,True]]))>0
    assert st['rear_prep_paid'].sum()<=1
    ns['_reset'](st,torch.tensor([True]));assert not st['rear_prep_initialized'].any() and not st['rear_prep_paid'].any()
    out['rear_prep_loaded_roll_support_flip_retry_wrong_side_reset']=True
    # Restore from paired stepping, then revoke a later same-tread catch-up.
    e,tick,land=make();e.scene['robot'].data.body_pos_w[0,:2,0]=1.2
    e.scene['robot'].data.body_pos_w[0,:2,2]=.89
    land(2,0);land(3,0);s=land(2,1)
    assert s['rear_recovery_event'] and not s['rear_event'].any()
    recovery=float(ns['stair_rear_transition_recovery'](e,**p)*e.step_dt)
    assert recovery==1
    s=land(3,1)
    refund=float(ns['stair_rear_style_refund'](e,**p)*e.step_dt)
    assert refund==ns['TUNING']['rear_recovery_weight']
    assert not tick()['rear_style_refund'].any()
    out['rear_repair_then_duplicate_net']=recovery*ns['TUNING']['rear_recovery_weight']-refund+ns['TUNING']['same_tread_event_weight']
    assert out['rear_repair_then_duplicate_net']<0
    # A true alternating transition also loses its style credit if its tread
    # later becomes shared; ordinary task crossing credit is untouched.
    e,tick,land=make();e.scene['robot'].data.body_pos_w[0,:2,0]=1.2
    e.scene['robot'].data.body_pos_w[0,:2,2]=.89
    land(2,0);s=land(3,1);assert s['rear_event'].any()
    s=land(2,1);assert s['rear_style_refund']==ns['TUNING']['rear_completion_weight']
    out['rear_delayed_duplicate_refund']=float(s['rear_style_refund'])
'''
assert s.count(needle)==1;s=s.replace(needle,add+needle);p.write_text(s)
