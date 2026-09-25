"""Exact uniform ratio-consistent Galerkin; CPU factors, one GPU layer at a time."""
from collections import OrderedDict
import math
import time
import torch
from torch.nn import functional as F
from utils.exact_layerwise_gram import LayerwiseIO, exact_selected_kernel_projections_fp32
from utils.layerwise_subcurvature import (
    parameter_columns_fp32,
    fixed_size_inclusion_probabilities,
    pivotal_sample,
)

def select_streamed_leverage(model, obs, actions, ratios, noise, seed, update,
                             sample_size=127, columns=2000, chunk=128):
    """True sampled-parameter-column leverage of the SAME shared sampled score.

    Only B x 2000 is materialized; ResNet factors are streamed in physical
    chunks.  Anchor selection changes; the full-RHS Galerkin solve is untouched.
    """
    device = actions.device
    B = len(actions)
    assert B == 16384 and sample_size == 127 and ratios.shape == (B,)
    parameters = [p for p in model.parameters() if p.requires_grad]
    parameter_count = sum(p.numel() for p in parameters)
    count = min(columns, parameter_count)
    generator = torch.Generator(device=device).manual_seed(
        seed * 1000003 + update * 101 + 128 * 17
    )
    flat_indices = torch.randperm(parameter_count, generator=generator, device=device)[:count]
    sketch = torch.empty((B, count), dtype=torch.float32, device=device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for start in range(0, B, chunk):
        stop = min(start + chunk, B)
        with torch.enable_grad():
            with LayerwiseIO(model) as io:
                values, logits = model(obs[start:stop].to(device))
                score = F.log_softmax(logits, dim=-1).gather(
                    1, actions[start:stop, None]
                ).squeeze(1) + 2 * noise[start:stop] * values
            factors = io.factors(score, retain_graph=False)
            io.assert_parameter_coverage()
            selected = parameter_columns_fp32(model, factors, flat_indices)
        sketch[start:stop] = (selected.detach() * ratios[start:stop, None].sqrt())
        del factors, selected, score, logits, values, io
    with torch.no_grad():
        left, _, _ = torch.pca_lowrank(
            sketch, q=sample_size + 32, center=False, niter=1
        )
        leverage = left[:, :sample_size].square().sum(dim=1)
        if not torch.isfinite(leverage).all() or leverage.sum() <= 0:
            raise FloatingPointError('nonfinite or zero leverage')
        probability = .95 * leverage / leverage.sum() + .05 / B
        inclusion = fixed_size_inclusion_probabilities(probability, sample_size)
        anchors = pivotal_sample(inclusion)
        if anchors.numel() != sample_size or torch.unique(anchors).numel() != sample_size:
            raise RuntimeError('leverage anchors are not exact-size unique')
        torch.cuda.synchronize(device)
        info = dict(mode='true2000column_pca_pivotal_no_ht',
                    leverage_seconds=time.perf_counter() - started,
                    leverage_probability_ess=float(probability.square().sum().reciprocal()),
                    leverage_probability_entropy_fraction=float(
                        (-(probability * probability.log()).sum() / math.log(B))
                    ),
                    selected_inclusion_min=float(inclusion[anchors].min()),
                    selected_inclusion_max=float(inclusion[anchors].max()),
                    parameter_columns=count, rank=sample_size,
                    uniform_mixture=.05)
    return anchors, info

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
