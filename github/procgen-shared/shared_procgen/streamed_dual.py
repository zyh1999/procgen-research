"""Exact uniform ratio-consistent Galerkin; CPU factors, one GPU layer at a time."""
from collections import OrderedDict
import torch
from torch.nn import functional as F
from .algebra import LayerwiseIO, exact_selected_kernel_projections_fp32

def named(model,grads):
    return OrderedDict((n,torch.zeros_like(p) if g is None else g.detach()) for (n,p),g in zip(model.named_parameters(),grads))

@torch.no_grad()
def solve_statistics(K,cross,norm2,targets,ratios,anchors,energy,divisor,damping):
    """Same weighted Schur formula as Task263; selected rows, not full scores."""
    K=K.double()/divisor; result=[];diagnostics=[]
    for k in range(targets.shape[1]):
        r=ratios[anchors,k].double().sqrt();h=r*targets[anchors,k].double()
        A=K*r[:,None]*r[None,:];A=(A+A.T)*.5
        A.diagonal().add_(damping)
        L,info=torch.linalg.cholesky_ex(A)
        if int(info)!=0:raise RuntimeError(f'Cholesky failure {int(info)}')
        E=energy[k].double()
        c=r*cross[:,k]/divisor;C=norm2[k]/divisor+damping*E
        uv=torch.cholesky_solve(torch.stack([h,c],dim=1),L);u,v=uv[:,0],uv[:,1]
        numerator=E-c@u;denominator=C-c@v
        if E==0:
            rho=torch.zeros_like(E);y=u
        else:
            if not torch.isfinite(denominator) or denominator<=0:raise FloatingPointError('nonpositive/nonfinite Schur denominator')
            rho=numerator/denominator;y=u-rho*v
        coeff=rho*targets[:,k].double();coeff[anchors]=y/r
        residual_top=A@y+c*rho-h
        residual_bottom=c@y+C*rho-E
        residual=torch.sqrt(residual_top.square().sum()+residual_bottom.square())/torch.sqrt(h.square().sum()+E.square()).clamp_min(1e-30)
        if not torch.isfinite(coeff).all() or not torch.isfinite(residual):raise FloatingPointError('nonfinite coefficient/residual')
        result.append(coeff.float())
        diagnostics.append(dict(residual=float(residual),rho=float(rho),numerator=float(numerator),denominator=float(denominator),energy=float(E),cholesky_info=int(info)))
    return torch.stack(result,dim=1),diagnostics

def solve_streamed(model,obs,actions,targets,ratios,noise,anchors,shared,divisor,chunk=128,damping=.5):
    """All non-anchor samples enter base and energy; no BxP materialization."""
    device=actions.device;B=len(actions);M=targets.shape[1]
    geometry=model if shared else model.pi_net
    trainable=[(n,p) for n,p in geometry.named_parameters() if p.requires_grad]
    parameters=[p for _,p in trainable]
    bases=[OrderedDict((n,torch.zeros_like(p)) for n,p in trainable) for _ in range(M)]
    tail=torch.ones(B,device=device,dtype=torch.bool);tail[anchors]=False
    weights=(ratios*targets)*tail[:,None]
    energy=(ratios.double()*targets.double().square()*tail[:,None]).sum(0)
    def score(indices,collect=False):
        x=obs[indices.cpu()].to(device)
        if collect:
            with LayerwiseIO(geometry) as io:
                if shared:values,logits=model(x)
                else:logits=model.forward_pi(x)
        else:
            if shared:values,logits=model(x)
            else:logits=model.forward_pi(x)
        z=F.log_softmax(logits,dim=-1).gather(1,actions[indices,None]).squeeze(1)
        if shared:z=z+2*noise[indices]*values
        return (z,io) if collect else z
    # Tail parameter-space sums stream through the ordinary backward engine.
    for start in range(0,B,chunk):
        idx=torch.arange(start,min(start+chunk,B),device=device)
        z=score(idx)
        for k in range(M):
            grads=torch.autograd.grad((z*weights[idx,k]).sum(),parameters,retain_graph=k<M-1,allow_unused=True)
            for (name,_),g in zip(trainable,grads):
                if g is not None:bases[k][name].add_(g.detach())
    # Only anchor d_in/d_out survive, on CPU. No full-batch factor storage.
    saved=OrderedDict()
    for start in range(0,len(anchors),chunk):
        idx=anchors[start:start+chunk];z,io=score(idx,True)
        factors=io.factors(z,retain_graph=False);io.assert_parameter_coverage()
        for name,(module,x,d) in factors.items():
            if name not in saved:saved[name]=(module,[],[])
            saved[name][1].append(x.detach().cpu())
            saved[name][2].append(None if d is None else d.detach().cpu())
        del io,factors,z
    q=len(anchors)
    K=torch.zeros((q,q),device=device,dtype=torch.float32)
    cross=torch.zeros((q,M),device=device,dtype=torch.float64)
    for name,(module,xs,ds) in saved.items():
        if ds[0] is None:continue
        factors=OrderedDict([(name,(module,torch.cat(xs).to(device),torch.cat(ds).to(device)))])
        ki,ci=exact_selected_kernel_projections_fp32(factors,bases)
        K.add_(ki);cross.add_(ci)
        del factors,ki,ci
    norm2=torch.stack([sum(x.double().square().sum() for x in b.values()) for b in bases])
    return solve_statistics(K,cross,norm2,targets,ratios,anchors,energy,divisor,damping)
