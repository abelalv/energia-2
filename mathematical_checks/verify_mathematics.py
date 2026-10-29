"""Small CPU/float64 checks only; no neural-network training or external imports.
Run from any directory: python3 verify_mathematics.py
Outputs beside this script. Random tests supplement, never replace, proofs.
"""
from pathlib import Path
import json, platform, sys
import numpy as np
from scipy.optimize import minimize

rng = np.random.default_rng(20261013)
checks = []
def check(name, condition, details):
    if not bool(condition):
        raise AssertionError(f'{name}: {details}')
    checks.append({'check': name, 'passed': True, 'details': details})
def nuc(a): return float(np.linalg.svd(a, compute_uv=False).sum())
def omega(ws): return 2*sum(nuc(w) for w in ws)
def fnorm(ws): return float(np.sqrt(sum(np.sum(w*w) for w in ws)))
def proj(ws, tau, method='sort'):
    if tau < 0: raise ValueError('negative budget')
    svds = [np.linalg.svd(w, full_matrices=False) for w in ws]
    sig = np.concatenate([s for q,s,v in svds])
    if tau == 0: return [np.zeros_like(w) for w in ws], float(sig.max(initial=0))
    if sig.sum() <= tau/2: return [w.copy() for w in ws], 0.
    if method == 'sort':
        a=np.sort(sig)[::-1]; c=np.cumsum(a); j=np.arange(1,len(a)+1)
        ix=np.flatnonzero(a>(c-tau/2)/j)[-1]
        t=float((c[ix]-tau/2)/(ix+1))
    else:
        lo,hi=0.,float(sig.max())
        for _ in range(100):
            t=(lo+hi)/2
            if np.maximum(sig-t,0).sum()>tau/2: lo=t
            else: hi=t
        t=hi
    return [(q*np.maximum(s-t,0))@v for q,s,v in svds],t

def adjacency(ws):
    dims=[ws[0].shape[1]]+[w.shape[0] for w in ws]
    off=np.cumsum([0]+dims); a=np.zeros((off[-1],off[-1]))
    for k,w in enumerate(ws):
        a[off[k+1]:off[k+2],off[k]:off[k+1]]=w
        a[off[k]:off[k+1],off[k+1]:off[k+2]]=w.T
    return a

path=[np.ones((1,1)), np.ones((1,1))]
e=nuc(adjacency(path))
check('graph_counterexample', np.isclose(e,2*np.sqrt(2)) and omega(path)==4, {'E':e,'Omega':omega(path)})
ratios=[]
for _ in range(100):
    dims=rng.integers(1,6,size=rng.integers(2,8)).tolist()
    ws=[rng.normal(size=(b,a)) for a,b in zip(dims,dims[1:])]
    E=nuc(adjacency(ws)); O=omega(ws);ratios.append(E/O)
    check(f'graph_bounds_{_}', O/2-1e-10<=E<=O+1e-10,{'ratio':E/O})
check('graph_zero',nuc(adjacency([np.zeros((2,3))]))==0, {})
for i,ws in enumerate([[np.diag([3.,3.,1.]),np.zeros((2,3))],
                       [np.ones((1,1)),np.diag([.1,.05])],
                       [rng.normal(size=(2,4)),rng.normal(size=(3,2))]]):
    for fraction in [0.,.01,.4,1.,1.2]:
        tau=fraction*omega(ws); z,t=proj(ws,tau); z2,t2=proj(ws,tau,'bisect')
        zp,_=proj(z,tau)
        check(f'projection_{i}_{fraction}', omega(z)<=tau+1e-10 and fnorm([a-b for a,b in zip(z,zp)])<1e-10 and fnorm([a-b for a,b in zip(z,z2)])<1e-10, {'tau':tau,'OmegaZ':omega(z),'threshold':t})
        if fraction>=1: check(f'interior_{i}_{fraction}',fnorm([a-b for a,b in zip(ws,z)])<1e-10,{})
        # Independent constrained quadratic program on singular values (SLSQP).
        sig=np.concatenate([np.linalg.svd(w,compute_uv=False) for w in ws])
        if tau>0:
            sol=minimize(lambda x:.5*np.sum((x-sig)**2),np.zeros_like(sig),jac=lambda x:x-sig,
                         bounds=[(0,None)]*len(sig),constraints=[{'type':'ineq','fun':lambda x:tau/2-x.sum(),'jac':lambda x:-np.ones_like(x)}],
                         method='SLSQP',options={'ftol':1e-12,'maxiter':1000})
            check(f'independent_qp_{i}_{fraction}',sol.success and np.max(np.abs(sol.x-np.maximum(sig-t,0)))<1e-7,{'status':str(sol.message)})
check('spectral_projection', np.sum((np.diag([2.,1.])-np.eye(2))**2)==1 and np.sum((np.diag([2.,1.])-np.diag([1.,.5]))**2)==1.25, {'clip_distance_squared':1.,'rescale_distance_squared':1.25})
z,_=proj([np.diag([2.,1.])],.02)
check('positive_radius_nonzero',nuc(z[0])>0,{'OmegaZ':omega(z)})
z,t=proj([np.diag([4.,3.]),np.diag([.1,.05])],2.)
check('joint_can_zero_layer',nuc(z[1])==0,{'threshold':t})
# Quadratic sharpness: W*=0 and equal singular perturbations across layers.
ws=[np.eye(2)*.3,np.eye(3)*.3];gamma=2.;R=5;gap=gamma/2*fnorm(ws)**2
check('qg_sharpness',np.isclose(omega(ws),2*np.sqrt(2*R/gamma)*np.sqrt(gap)),{'lhs':omega(ws),'rhs':2*np.sqrt(2*R/gamma)*np.sqrt(gap)})
# QG with nonunique minima and explicit projection: f=.5||B(theta-a)||².
B=np.array([[2.,0.,0.],[0.,1.,0.]]);a=np.array([1.,2.,3.]);theta=np.array([1.3,1.7,8.]);plus=theta-np.linalg.pinv(B)@(B@(theta-a))
gap=.5*np.sum((B@(theta-a))**2);gamma=1.
check('qg_nonunique',gap>=gamma/2*np.sum((theta-plus)**2)-1e-12 and np.allclose(plus,[1.,2.,8.]),{'projection':plus.tolist(),'gap':gap})
for i in range(50):
    ws=[rng.normal(size=(2,3)),rng.normal(size=(1,2))];tau=.7*omega(ws)
    z,_=proj([w+.1*rng.normal(size=w.shape) for w in ws],tau)
    residual=[w-v for w,v in zip(ws,z)];D=3
    v=max(0,omega(ws)-tau);rhs=max(0,omega(z)-tau)+2*np.sqrt(D)*fnorm(residual)
    check(f'feasibility_{i}',v<=rhs+1e-10,{'violation':v,'bound':rhs})
# Full-space stationarity identity; f=.5||W-H||²+.5||b-c||², L=1.
W=[rng.normal(size=(2,2))];H=[rng.normal(size=(2,2))];old=[rng.normal(size=(2,2))];u=[rng.normal(size=(2,2))]
rho=1.7;tau=1.;Z,_=proj([W[0]+u[0]],tau);r=W[0]-Z[0];s=rho*(Z[0]-old[0]);y=rho*(u[0]+r)
b=np.array([.2,-.3]);c=np.array([.7,.1]);eb=b-c;ew=W[0]-H[0]+rho*(W[0]-old[0]+u[0])
station_candidate=np.sqrt(np.sum((Z[0]-H[0]+y)**2)+np.sum(eb**2))
bound=np.sqrt(np.sum(ew**2)+np.sum(eb**2))+np.linalg.norm(s)+np.linalg.norm(r)
check('stationarity_identity', np.allclose(W[0]-H[0]+y,ew-s) and station_candidate<=bound+1e-10,{'candidate_norm':station_candidate,'upper_bound':bound})
for i in range(30):
    V,_=proj([rng.normal(size=(2,2))],tau)
    check(f'projection_normal_{i}', np.sum(y*(V[0]-Z[0]))<=1e-10,{})
# Correct dual increment.
f=2.;r=np.array([.7,-.2]);u=np.array([.1,.4]);rho=3.
L=lambda v:f+rho/2*np.sum((r+v)**2)-rho/2*np.sum(v*v)
check('dual_increment',np.isclose(L(u+r)-L(u),rho*np.sum(r*r)),{'increment':L(u+r)-L(u),'rho_r_squared':rho*np.sum(r*r)})
F=lambda w:.5*(w-2)**2+max(0,2*abs(w)-2)
check('penalty_counterexample',F(1)==.5 and F(1.5)-F(1)==.625 and 1>.625,{'Fstar':F(1),'gap':F(1.5)-F(1),'violation':1})
check('multiplier_nonconvex_counterexample',2>1/1.5,{'lhs':2,'rhs':1/1.5})
# Summable polynomial error is not geometric: delta_k >= epsilon_(k-1).
delta=0.;q=.5
for k in range(1,201): delta=q*delta+1/k**2
check('summable_error_not_geometric',delta>=1/200**2 and delta/q**200>1e40,{'delta200':delta,'delta_over_q200':delta/q**200})
W1=rng.normal(size=(3,2));W2=rng.normal(size=(1,3));X=rng.normal(size=(2,7));c=3.
out=W2@np.maximum(W1@X,0);scaled=(W2/c)@np.maximum((c*W1)@X,0)
check('relu_scaling',np.allclose(out,scaled) and not np.isclose(omega([W1,W2]),omega([c*W1,W2/c])),{'prediction_max_error':float(np.max(np.abs(out-scaled))),'Omega_before':omega([W1,W2]),'Omega_after':omega([c*W1,W2/c])})
sig=np.array([3.,2.,.1]);p=sig/sig.sum();p2=(sig/3)/(sig/3).sum()
check('effective_rank_invariant',np.allclose(p,p2),{'rank':float(np.exp(-np.sum(p*np.log(p))))})
result={'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),'seed':20261013,'dtype':'float64','training_runs':0,'passed':len(checks),'graph_ratio_range':[min(ratios),max(ratios)],'checks':checks}
path=Path(__file__).with_name('current_results.json');path.write_text(json.dumps(result,indent=2))
print(json.dumps({k:result[k] for k in ['python','numpy','seed','dtype','training_runs','passed','graph_ratio_range']},indent=2))
