"""Task266: true 16384 minibatch updates, streaming physical chunks of128."""
import argparse,csv,json,math,os,time,types
from collections import deque
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from procgen import ProcgenEnv
from vec_env import VecExtractDictObs,VecMonitor,VecNormalize
from utils.utils import ActorCritic,SharedActorCritic,build_resnet,set_seed
from utils.runners import Runner
from streamed_dual import solve_streamed, select_streamed_leverage

ROLLOUT=131072;BATCH=16384;CHUNK=128;EPOCHS=4;ENDPOINT=10092544
ENVS=['bigfish','bossfight','caveflyer','coinrun','starpilot','jumper','maze','miner']
ARMS=['shared_dual128','shared_dual4096','shared_ppo','noshared_dual128','noshared_dual4096','noshared_ppo']

def slices(n):
    for start in range(0,n,CHUNK):yield slice(start,min(n,start+CHUNK))

def categorical(logits,actions):
    lp=F.log_softmax(logits,dim=-1)
    return lp,lp.gather(1,actions[:,None]).squeeze(1)

def train(arm,environment,seed,root):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    (root/'trainer_pid').write_text(str(os.getpid()));(root/'status').write_text('STARTING')
    set_seed(seed);torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda:0');shared=arm.startswith('shared_');ppo=arm.endswith('_ppo')
    q=0 if ppo else (128 if arm.endswith('128') else 4096)
    if shared:lr,lo,hi,lrmin,lrmax=.5,.005,.04,.0001,.5
    elif environment=='maze':lr,lo,hi,lrmin,lrmax=.0025,.0001,.008,.00025,.015
    else:lr,lo,hi,lrmin,lrmax=.005,.0005,.0025,.0001,.05
    meta=dict(task='PROCGEN-LARGEBATCH-SHARED-NOSHARED-DUAL128-DUAL4096-PPO-8ENV-S0TO4-20M-BEDE-20260924-266',arm=arm,environment=environment,seed=seed,shared=shared,rollout=ROLLOUT,minibatch=BATCH,epochs=4,physical_chunk=CHUNK,curvature_dimension=q,anchors=max(0,q-1),normalization_divisor=q,damping=None if ppo else .5,requested_transitions=20000000,endpoint=ENDPOINT,entropy=0,network='ResNet[8,16],hidden256',num_envs=16,nsteps=8192,levels=10,mode='easy',popart=not ppo,critic_ratio=False,kl_band=None if ppo else [lo,hi],lr=.0003 if ppo else lr,lr_range=None if ppo else [lrmin,lrmax],controller='none' if ppo else 'per_minibatch_postupdate_behavior_kl',history=False,full_rhs_samples=BATCH,hardware=torch.cuda.get_device_name(0),slurm_job=os.environ.get('SLURM_JOB_ID'))
    cap_multiplier=int(os.environ['TASK267_CAP_MULTIPLIER'])
    assert arm=='shared_dual128' and cap_multiplier==3
    original_cap=lrmax;lrmax*=cap_multiplier
    meta.update(task='PROCGEN-LARGEBATCH-SHARED-DUAL127P1-LEVERAGE2000-CAPX3-FIRST4-S012-10M-20260924',method='shared_dual127p1_true2000column_leverage_pca_pivotal_no_ht',anchor_sampling='leverage',leverage_parameter_columns=2000,leverage_rank=127,leverage_oversample=32,leverage_power_iterations=1,leverage_uniform_mixture=.05,lr_cap_multiplier=cap_multiplier,original_lr_cap=original_cap,lr_range=[lrmin,lrmax],requested_transitions=10000000,endpoint=ENDPOINT)
    (root/'metadata.json').write_text(json.dumps(meta,indent=2))
    v=ProcgenEnv(num_envs=16,env_name=environment,num_levels=10,start_level=0,distribution_mode='easy',rand_seed=seed)
    v=VecExtractDictObs(v,'rgb');v=VecMonitor(v,filename=str(root))
    fn,preprocess=build_resnet(64,256,device=device,with_bn=False,depths=[8,16])
    cfg=types.SimpleNamespace(norm_obs=False,dropout=0.,hidden_size=256,a_dropout=0.,c_dropout=0.,a_hidden_size=256,c_hidden_size=256,a_num_layers=2,c_num_layers=2)
    model=(SharedActorCritic if shared else ActorCritic)(fn,(3,64,64),cfg,n_actions=v.action_space.n,with_popart=not ppo,device=device).to(device)
    v=VecNormalize(v,norm_ret=False,obs_preprocess=preprocess)
    runner=Runner(env=v,model=model,nsteps=8192,gamma=.999,lam=.95,adv_type='gae',device=device)
    geometry=model if shared else model.pi_net
    optimizer=torch.optim.Adam(geometry.parameters(),lr=3e-4,eps=1e-5) if ppo else torch.optim.SGD(geometry.parameters(),lr=lr,momentum=0.)
    critic=None if shared else torch.optim.Adam(model.v_net.parameters(),lr=3e-4 if ppo else .001,eps=1e-5 if ppo else 1e-8)
    wandb_run=None
    try:
        import wandb
        wandb_run=wandb.init(entity='zhouyihe',project='procgen-largebatch-ef-leverage2000',name=f'{arm}-leverage2000-capx3-{environment}-s{seed}',config=meta,dir=str(root))
        (root/'wandb_url').write_text(wandb_run.url)
    except Exception as e:(root/'wandb_error').write_text(repr(e))
    trace=(root/'metric_trace.jsonl').open('w',buffering=1)
    progress=(root/'progress.csv').open('w',buffering=1);writer=None
    epinfo=deque(maxlen=100);start_time=time.time();updates=0
    (root/'status').write_text('RUNNING')
    for rollout in range(ENDPOINT//ROLLOUT):
        start=time.time();model.eval()
        obs,ret,actions,advantages,old,episodes=runner.run();epinfo.extend(episodes)
        assert obs.device.type=='cpu' and len(obs)==ROLLOUT
        if not ppo:
            model.last_v_layer.update(ret);ret=model.last_v_layer.normalize(ret);advantages=model.last_v_layer.normalize(advantages)
        model.train();indices=np.arange(ROLLOUT);metrics=[]
        for epoch in range(EPOCHS):
            np.random.shuffle(indices)
            for offset in range(0,ROLLOUT,BATCH):
                ids=indices[offset:offset+BATCH];x=obs[ids];act=actions[ids];adv=advantages[ids];target=ret[ids];oldlp=F.log_softmax(old[ids],dim=-1)
                adv=adv-adv.mean();adv=adv/(adv.std()+1e-8) if ppo else adv/(adv.square().mean().sqrt()+1e-8)
                before=optimizer.param_groups[0]['lr'];ratios=[]
                with torch.no_grad():
                    for sl in slices(BATCH):
                        logits=model.forward_pi(x[sl].to(device));lp,logp=categorical(logits,act[sl])
                        ratios.append((logp-oldlp[sl].gather(1,act[sl,None]).squeeze(1)).exp())
                ratio=torch.cat(ratios).clamp(.1,10)
                diagnostics=[]
                if not ppo:
                    noise=torch.randn(BATCH,device=device) if shared else torch.zeros(BATCH,device=device)
                    anchors,selection=select_streamed_leverage(model,x,act,ratio,noise,seed,updates,sample_size=q-1,columns=2000,chunk=CHUNK)
                    targets=torch.stack([adv,torch.ones_like(adv)],1) if shared else adv[:,None]
                    weights=torch.stack([ratio,torch.ones_like(ratio)],1) if shared else ratio[:,None]
                    coeff,diagnostics=solve_streamed(model,x,act,targets,weights,noise,anchors,shared,q,chunk=CHUNK)
                optimizer.zero_grad(set_to_none=True);loss_pi_total=loss_v_total=0.
                for sl in slices(BATCH):
                    if shared:values,logits=model(x[sl].to(device))
                    else:logits=model.forward_pi(x[sl].to(device))
                    lp,logp=categorical(logits,act[sl])
                    if ppo:
                        r=(logp-oldlp[sl].gather(1,act[sl,None]).squeeze(1)).exp()
                        pl=torch.maximum(-r*adv[sl],-r.clamp(.8,1.2)*adv[sl]).sum()/BATCH
                        vl=.5*(values-target[sl]).square().sum()/BATCH if shared else pl.new_zeros(())
                    else:
                        if shared:
                            current_ratio=(logp-oldlp[sl].gather(1,act[sl,None]).squeeze(1)).exp().clamp(.1,10)
                            pl=-(current_ratio*coeff[sl,0].detach()).sum()/BATCH
                        else:
                            pl=-(ratio[sl].detach()*logp*coeff[sl,0].detach()).sum()/BATCH
                        vl=((values-target[sl]).square()*coeff[sl,1].detach()).sum()/BATCH if shared else pl.new_zeros(())
                    (pl+vl).backward();loss_pi_total+=float(pl.detach());loss_v_total+=float(vl.detach())
                gradnorm=torch.nn.utils.clip_grad_norm_(geometry.parameters(),.5,error_if_nonfinite=True)
                optimizer.step();kl=entropy=0.
                with torch.no_grad():
                    for sl in slices(BATCH):
                        lp=F.log_softmax(model.forward_pi(x[sl].to(device)),dim=-1)
                        kl+=float((oldlp[sl].exp()*(oldlp[sl]-lp)).sum()/BATCH)
                        entropy+=float(-(lp.exp()*lp).sum()/BATCH)
                if not math.isfinite(kl):raise FloatingPointError('nonfinite postupdate KL')
                if not ppo:
                    after=max(before/1.5,lrmin) if kl>hi else min(before*1.5,lrmax) if kl<lo else before
                    optimizer.param_groups[0]['lr']=after
                else:after=before
                updates+=1
                item=dict(rollout=rollout,epoch=epoch,minibatch=offset//BATCH,update=updates,steps=(rollout+1)*ROLLOUT,kl=kl,entropy=entropy,lr_before=before,lr_after=after,loss_pi=loss_pi_total,loss_v=loss_v_total,gradient_norm=float(gradnorm),q=q,divisor=q,full_rhs_samples=BATCH,selection=selection,diagnostics=diagnostics)
                trace.write(json.dumps(item)+'\n');metrics.append(item)
        if not shared:
            for epoch in range(EPOCHS):
                np.random.shuffle(indices)
                for offset in range(0,ROLLOUT,BATCH):
                    ids=indices[offset:offset+BATCH];x=obs[ids];target=ret[ids];critic.zero_grad(set_to_none=True)
                    for sl in slices(BATCH):
                        loss=(model.forward_v(x[sl].to(device))-target[sl]).square().sum()/BATCH
                        if ppo:loss=loss*.5
                        loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.v_net.parameters(),.5 if ppo else 5.,error_if_nonfinite=True);critic.step()
        assert len(metrics)==32
        row={'misc/total_timesteps':(rollout+1)*ROLLOUT,'misc/time_elapsed':time.time()-start_time,'fps':ROLLOUT/(time.time()-start),'eprewmean':float(np.mean([e['r'] for e in epinfo])) if epinfo else float('nan'),'entropy':metrics[-1]['entropy'],'kl':metrics[-1]['kl'],'lr':optimizer.param_groups[0]['lr'],'updates':updates,'solve_residual_max':max([d['residual'] for m in metrics for d in m['diagnostics']] or [0.]),'curvature_dimension':q,'system_divisor':q,'actor_updates_per_rollout':len(metrics),'leverage_seconds_per_update':float(np.mean([m['selection']['leverage_seconds'] for m in metrics])),'leverage_probability_ess_last':metrics[-1]['selection']['leverage_probability_ess']}
        if writer is None:writer=csv.DictWriter(progress,fieldnames=list(row));writer.writeheader()
        writer.writerow(row);print(json.dumps(row),flush=True)
        if wandb_run:wandb_run.log(row,step=row['misc/total_timesteps'])
        temp=root/'checkpoint.tmp';torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'critic_optimizer':None if critic is None else critic.state_dict(),'rollout':rollout},temp);temp.replace(root/'model.ckpt')
    trace.close();progress.close();(root/'rc').write_text('0');(root/'status').write_text('PASS')
    if wandb_run:wandb_run.finish()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--environment',choices=ENVS,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--root',required=True);a=p.parse_args()
    try:train(a.arm,a.environment,a.seed,a.root)
    except Exception as e:
        r=Path(a.root)
        if r.exists():(r/'rc').write_text('1');(r/'status').write_text('FAIL');(r/'error').write_text(repr(e))
        raise
