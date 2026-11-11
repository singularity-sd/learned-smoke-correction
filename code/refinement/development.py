import importlib.util
import json
import math
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from scipy import fft

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("retry_smoke",ROOT.parent/"retry/experiment.py")
retry=importlib.util.module_from_spec(spec)
spec.loader.exec_module(retry)
common=retry.common
DT=common.DT
HIGH=common.HIGH
FRAMES=common.FRAMES
FIT=range(1000,1048)
SETUPS=tuple((n,interp,temporal) for n in (64,80,96)
             for interp in ("linear","cubic") for temporal in (False,True))
BASELINES=tuple((f"maccormack{n}","maccormack",n,False,0.) for n in (48,64,80,96,144))
BASELINES+=tuple((f"euler_{interp}{n}",interp,n,False,0.) for n in (64,80,96,144) for interp in ("linear","cubic"))


def variant_name(n,interp,temporal,gain):
    return f"{'time' if temporal else 'mid'}_{interp}{n}_{gain:g}"


class MidStep(retry.FastStep):
    def __init__(self,n,interp,temporal):
        super().__init__(n,cv2.INTER_CUBIC if interp=="cubic" else cv2.INTER_LINEAR)
        self.temporal=temporal
        self.previous=None
        self.shape=(n,n)
        self.mode_basis=self.mode_basis[:4]

    def velocity(self,omega,wind):
        psi=fft.rfft2(omega)*self.inverse
        vel=np.empty((*self.shape,2),np.float32)
        vel[:,:,0]=fft.irfft2(psi*self.ky,s=self.shape)+wind
        vel[:,:,1]=-fft.irfft2(psi*self.kx,s=self.shape)
        return vel

    def advance_spectrum(self,state,s,t,features):
        current=self.velocity(state[1],s["wind"])
        direction=current
        if self.temporal and self.previous is not None:
            direction=1.5*current-.5*self.previous
        self.previous=current
        scale=self.dt*self.n
        mid=cv2.remap(direction,self.ix-.5*scale*direction[:,:,0],
                      self.iy-.5*scale*direction[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_WRAP)
        dx,dy=scale*mid[:,:,0],scale*mid[:,:,1]
        fields=cv2.remap(np.ascontiguousarray(state.transpose(1,2,0)),self.ix-dx,self.iy-dy,
                         self.interpolation,borderMode=cv2.BORDER_WRAP)
        rho=fields[:,:,0]
        if self.interpolation==cv2.INTER_CUBIC:
            rho=np.clip(rho,0,state[0].max())
        rho=rho*self.rho_decay+self.dt*self.source(s,t+.5*self.dt)
        omega=fields[:,:,1]+.5*self.n*self.dt*s["beta"]*(np.roll(rho,-1,axis=1)-np.roll(rho,1,axis=1))
        spectrum=fft.rfft2(np.stack((rho,omega)))*self.diffusion
        if not features:
            return spectrum,None
        fx,fy=dx-np.floor(dx),dy-np.floor(dy)
        bx,by=float(np.mean(4*fx*(1-fx))),float(np.mean(4*fy*(1-fy)))
        return spectrum,np.array([bx,by,bx,by],dtype=np.float32)

    def frame(self,state,s,t,weights=None,gain=1.):
        spectrum,factors=self.advance_spectrum(state,s,t,weights is not None)
        if weights is not None:
            response=np.einsum("cj,jxy->cxy",weights*factors[None,:]*gain,self.mode_basis)
            spectrum*=1+np.clip(response,-.18,.55)
        out=fft.irfft2(spectrum,s=self.shape)
        out[0]=np.maximum(out[0],0)
        out[1]-=out[1].mean()
        return out


def fit():
    solvers={k:MidStep(*k) for k in SETUPS}
    grams={k:np.zeros((2,4,4),dtype=np.float64) for k in SETUPS}
    rhs={k:np.zeros((2,4),dtype=np.float64) for k in SETUPS}
    for i,seed in enumerate(FIT):
        s=common.scene(seed)
        ref=common.ref_scene(s)
        coarse={n:{t:retry.resize(ref[t],n) for t in range(FRAMES+1) if t%5 in (0,1,4)}
                for n in (64,80,96)}
        for k,solver in solvers.items():
            n,interp,temporal=k
            basis=solver.mode_basis.reshape(4,-1).astype(np.float64)
            w=(.15+solver.sx+solver.sy).reshape(-1)
            for t in range(0,FRAMES,5):
                state=coarse[n][t]
                target=coarse[n][t+1]
                solver.previous=solver.velocity(coarse[n][t-1][1],s["wind"]) if temporal and t else None
                pred,factors=solver.advance_spectrum(state,s,t*DT,True)
                wanted=fft.rfft2(target)
                design=basis*factors[:,None]
                for c in range(2):
                    field=pred[c].reshape(-1)
                    err=(wanted[c]-pred[c]).reshape(-1)
                    grams[k][c]+=(design*(w*np.abs(field)**2))@design.T
                    rhs[k][c]+=design@(w*np.real(np.conj(field)*err))
        if (i+1)%12==0:
            print("fit scenes",i+1,"/",len(FIT),flush=True)
    output={}
    for k in SETUPS:
        penalty=.001*np.trace(grams[k],axis1=1,axis2=2)/4
        output[k]=np.stack([np.linalg.solve(grams[k][c]+penalty[c]*np.eye(4),rhs[k][c])
                             for c in range(2)]).astype(np.float32)
    (ROOT/"results/weights.json").write_text(json.dumps(
        [{"setup":k,"weights":v.tolist()} for k,v in output.items()],indent=2))
    return output


def simulate(s,entry,weights):
    name,kind,n,temporal,gain=entry
    if kind in ("linear","cubic") and name.startswith(("mid_","time_")):
        solver=MidStep(n,kind,temporal)
        w=weights[(n,kind,temporal)] if gain else None
        state=solver.initial(s)
        frames=[]
        durations=[]
        for i in range(FRAMES):
            start=perf_counter()
            state=solver.frame(state,s,i*DT,w,gain)
            frames.append(common.render_state(state))
            durations.append(perf_counter()-start)
        output=np.array(frames)
        assert np.isfinite(output).all()
        return output,durations
    return retry.simulate(s,kind,n)


def table_on(seeds,entries,weights,*,gust=False):
    scores={e[0]:[] for e in entries}
    times={e[0]:[] for e in entries}
    rng=np.random.default_rng(531)
    for j,seed in enumerate(seeds):
        s=common.scene(seed,gust=gust and j>=len(seeds)//2)
        ref=common.ref_scene(s)[1:,0]
        for i in rng.permutation(len(entries)):
            e=entries[i]
            pred,elapsed=simulate(s,e,weights)
            scores[e[0]].append(common.rmse(pred,ref))
            times[e[0]].extend(elapsed)
        if j%4==3:
            print("evaluated",j+1,"/",len(seeds),flush=True)
    return [dict(entry=e,rmse=float(np.mean(scores[e[0]])),
                 median_ms=float(1e3*np.median(times[e[0]]))) for e in entries]


def main():
    cv2.setNumThreads(1)
    weights=fit()
    entries=list(BASELINES)
    entries.extend((variant_name(n,i,t,gain),i,n,t,gain) for n,i,t in SETUPS for gain in (0.,.5,1.,1.5))
    table=table_on(common.VAL,entries,weights)
    classics=[r for r in table if r["entry"][-1]==0.]
    for row in table:
        refs=[r for r in classics if r["median_ms"]<=1.15*row["median_ms"]]
        if refs:
            best=min(refs,key=lambda r:r["rmse"])
            row["reference"]=best["entry"][0]
            row["reduction"]=1-row["rmse"]/best["rmse"]
    table.sort(key=lambda r:r.get("reduction",-1),reverse=True)
    (ROOT/"results/development.json").write_text(json.dumps(table,indent=2))
    print(json.dumps(table[:12],indent=2),flush=True)


if __name__=="__main__":
    main()
