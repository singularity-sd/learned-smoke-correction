import json
import math
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np
from scipy import fft

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
import experiment as common

DT=common.DT
HIGH=common.HIGH
FRAMES=common.FRAMES
FIT=range(1000,1064)
TEST=range(4000,4024)
GRIDS=(64,96)
CLASSICS=tuple((method,n) for method,ns in (("maccormack",(48,64,96,144)),
                                           ("linear",(64,96,144)),("cubic",(64,96,144)))
               for n in ns)


class FastStep(common.Solver):
    def __init__(self,n,interpolation):
        super().__init__(n,1)
        self.interpolation=interpolation
        sx=np.sin(np.pi*fft.rfftfreq(n)).astype(np.float32)[None,:]**2
        sy=np.sin(np.pi*fft.fftfreq(n)).astype(np.float32)[:,None]**2
        self.sx=np.broadcast_to(sx,(n,n//2+1)).copy()
        self.sy=np.broadcast_to(sy,(n,n//2+1)).copy()
        self.mode_basis=np.stack((self.sx,self.sy,self.sx**2,self.sy**2,self.sx,self.sy))
        self.diffusion=np.stack((self.diff_rho,self.diff_w))
        self.rho_decay=math.exp(-.16*self.dt)

    def spectrum(self,state,s,t,features=False):
        rho,w=state
        psi=fft.rfft2(w)*self.inverse
        u=fft.irfft2(psi*self.ky,s=(self.n,self.n))+s["wind"]
        v=-fft.irfft2(psi*self.kx,s=(self.n,self.n))
        dx,dy=self.dt*self.n*u,self.dt*self.n*v
        transported=cv2.remap(np.ascontiguousarray(state.transpose(1,2,0)),
                              self.ix-dx,self.iy-dy,self.interpolation,borderMode=cv2.BORDER_WRAP)
        dens=transported[:,:,0]
        if self.interpolation==cv2.INTER_CUBIC:
            dens=np.clip(dens,0,rho.max())
        dens=dens*self.rho_decay+self.dt*self.source(s,t+.5*self.dt)
        grad=.5*self.n*(np.roll(dens,-1,axis=1)-np.roll(dens,1,axis=1))
        omega=transported[:,:,1]+self.dt*s["beta"]*grad
        spec=fft.rfft2(np.stack((dens,omega)))*self.diffusion
        if not features:
            return spec,None
        fx,fy=dx-np.floor(dx),dy-np.floor(dy)
        bx,by=float(np.mean(4*fx*(1-fx))),float(np.mean(4*fy*(1-fy)))
        return spec,np.array([bx,by,bx,by,1,1],dtype=np.float32)

    def frame(self,state,s,t,weights=None,strength=1.):
        spec,condition=self.spectrum(state,s,t,weights is not None)
        if weights is not None:
            response=np.einsum("cj,jxy->cxy",weights*condition[None,:]*strength,self.mode_basis)
            spec*=1+np.clip(response,-.18,.55)
        out=fft.irfft2(spec,s=(self.n,self.n))
        out[0]=np.maximum(out[0],0)
        out[1]-=out[1].mean()
        return out


def resize(state,n):
    return np.stack([cv2.resize(f,(n,n),interpolation=cv2.INTER_AREA) for f in state])


def fit():
    solvers={n:FastStep(n,cv2.INTER_LINEAR) for n in GRIDS}
    gram={n:np.zeros((2,6,6),dtype=np.float64) for n in GRIDS}
    rhs={n:np.zeros((2,6),dtype=np.float64) for n in GRIDS}
    for i,seed in enumerate(FIT):
        s=common.scene(seed)
        ref=common.ref_scene(s)
        for n,solver in solvers.items():
            basis=solver.mode_basis.reshape(6,-1).astype(np.float64)
            w=(.15+solver.sx+solver.sy).reshape(-1)
            for t in range(0,FRAMES,4):
                state=resize(ref[t],n)
                target=resize(ref[t+1],n)
                pred,features=solver.spectrum(state,s,t*DT,True)
                target_spectrum=fft.rfft2(target)
                predictors=basis*features[:,None]
                for c in range(2):
                    field=pred[c].reshape(-1)
                    error=(target_spectrum[c]-pred[c]).reshape(-1)
                    power=w*np.abs(field)**2
                    gram[n][c]+=(predictors*power)@predictors.T
                    rhs[n][c]+=predictors@(w*np.real(np.conj(field)*error))
        if (i+1)%16==0:
            print("fit scenes",i+1,"/",len(FIT),flush=True)
    weights={}
    for n in GRIDS:
        penalty=.001*np.trace(gram[n],axis1=1,axis2=2)/6
        weights[n]=np.stack([np.linalg.solve(gram[n][c]+penalty[c]*np.eye(6),rhs[n][c])
                             for c in range(2)]).astype(np.float32)
    (ROOT/"results/weights.json").write_text(json.dumps({n:w.tolist() for n,w in weights.items()},indent=2))
    return weights


def simulate(s,method,n,weights=None,strength=1.):
    if method in ("linear","cubic","learned"):
        solver=FastStep(n,cv2.INTER_CUBIC if method=="cubic" else cv2.INTER_LINEAR)
    else:
        solver=common.Solver(n,4 if method=="reference" else 1)
    state=solver.initial(s)
    frames=[]
    latency=[]
    for frame in range(FRAMES):
        start=perf_counter()
        if method=="learned":
            state=solver.frame(state,s,frame*DT,weights,strength)
        else:
            state=solver.frame(state,s,frame*DT)
        image=common.render_state(state)
        latency.append(perf_counter()-start)
        frames.append(image)
    output=np.asarray(frames)
    assert np.isfinite(output).all()
    return output,latency


def choose(weights):
    methods=[("maccormack",96,0.)]+[("learned",n,s) for n in GRIDS for s in (.5,1.,1.5)]
    errors={m:[] for m in methods}
    times={m:[] for m in methods}
    rng=np.random.default_rng(832)
    for seed in common.VAL:
        s=common.scene(seed)
        ref=common.ref_scene(s)[1:,0]
        for i in rng.permutation(len(methods)):
            m=methods[i]
            frames,durations=simulate(s,m[0],m[1],weights.get(m[1]),m[2])
            errors[m].append(common.rmse(frames,ref))
            times[m].extend(durations)
    table=[dict(method=m[0],resolution=m[1],strength=m[2],rmse=float(np.mean(errors[m])),
                median_ms=float(1000*np.median(times[m]))) for m in methods]
    available=[row for row in table[1:] if row["median_ms"]<=table[0]["median_ms"]]
    chosen=min(available,key=lambda row:row["rmse"]) if available else min(table[1:],key=lambda row:row["median_ms"])
    (ROOT/"results/validation.json").write_text(json.dumps(dict(candidates=table,chosen=chosen),indent=2))
    print("Locked validation choice",chosen,flush=True)
    return chosen


def evaluate(weights,chosen):
    models={f"{m}{n}":(m,n,0.) for m,n in CLASSICS}
    models.update(learned=(chosen["method"],chosen["resolution"],chosen["strength"]),
                  reference=("reference",HIGH,0.))
    timings={key:[] for key in models}
    rows=[]
    rng=np.random.default_rng(4024)
    for method,n,s in models.values():
        simulate(common.scene(998),method,n,weights.get(n),s)
    for i,seed in enumerate(TEST):
        config=common.scene(seed,gust=i>=12)
        videos={}
        for key in rng.permutation(list(models)):
            method,n,strength=models[key]
            videos[key],elapsed=simulate(config,method,n,weights.get(n),strength)
            timings[key].extend(elapsed)
        ref=videos["reference"]
        metrics={key:dict(rmse=common.rmse(video,ref),gradient=common.grad_rmse(video,ref))
                 for key,video in videos.items()}
        rows.append(dict(seed=seed,strong_wind=i>=12,metrics=metrics))
        if i in (0,12):
            np.savez_compressed(ROOT/"results"/f"demo_{seed}.npz",**videos)
    summary={}
    for m in models:
        summary[m]=dict(rmse=float(np.mean([r["metrics"][m]["rmse"] for r in rows])),
                        gradient_rmse=float(np.mean([r["metrics"][m]["gradient"] for r in rows])),
                        median_ms=float(1000*np.median(timings[m])),
                        p95_ms=float(1000*np.quantile(timings[m],.95)),
                        normal_rmse=float(np.mean([r["metrics"][m]["rmse"] for r in rows if not r["strong_wind"]])),
                        strong_wind_rmse=float(np.mean([r["metrics"][m]["rmse"] for r in rows if r["strong_wind"]])))
    options=[key for key in models if key not in ("learned","reference") and
             summary[key]["median_ms"]<=1.15*summary["learned"]["median_ms"]]
    best=min(options,key=lambda m:summary[m]["rmse"]) if options else None
    draws=rng.integers(len(rows),size=(5000,len(rows)))
    differences={}
    for m in models:
        if m in ("learned","reference"):
            continue
        delta=np.array([r["metrics"][m]["rmse"]-r["metrics"]["learned"]["rmse"] for r in rows])
        differences[m]=dict(relative=float(delta.mean()/summary[m]["rmse"]),
                            ci95=np.quantile(delta[draws].mean(axis=1),[.025,.975]).tolist())
    speed=summary["reference"]["median_ms"]/summary["learned"]["median_ms"]
    hit=best is not None and differences[best]["relative"]>=.05 and differences[best]["ci95"][0]>0 and speed>=2
    report=dict(forecast="T3-2026-0032",scenes=len(rows),frames_per_scene=FRAMES,chosen=chosen,methods=summary,
                comparable_classics=options,best_same_cost=best,differences=differences,
                speed_vs_reference=speed,meets_gate=bool(hit))
    (ROOT/"results/scenes.json").write_text(json.dumps(rows,indent=2)+"\n")
    (ROOT/"results/timings.json").write_text(json.dumps(timings,indent=2)+"\n")
    (ROOT/"results.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    cv2.setNumThreads(1)
    weights=fit()
    evaluate(weights,choose(weights))
