import json
import math
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from scipy import fft
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent
LOW=48
HIGH=144
FRAMES=40
DT=.045
TRAIN=range(1000,1088)
VAL=range(2000,2012)
TEST=range(3000,3024)
EPOCHS=12
WS=8.
NAMES=("classic48","classic48x4","classic64","classic96","classic144","hybrid48","reference")


def scene(seed, gust=False):
    rng=np.random.default_rng(seed)
    wind=float(rng.uniform(-.21,.21))
    if gust:
        wind=float(rng.choice([-1,1])*rng.uniform(.23,.34))
    return dict(seed=seed,wind=wind,beta=float(rng.uniform(.8,1.5)),
                center=float(rng.uniform(.30,.70)), height=float(rng.uniform(.16,.27)),
                radius=float(rng.uniform(.022,.043)), drift=float(rng.uniform(.018,.11)),
                frequency=float(rng.uniform(.65,1.8)), phase=float(rng.uniform(0,6.28)),
                power=float(rng.uniform(3.0,5.5)),
                vortices=[(float(rng.uniform(.16,.84)),float(rng.uniform(.18,.80)),
                           float(rng.uniform(.075,.14)),float(rng.uniform(-2.8,2.8))) for _ in range(5)])


def periodic_delta(x,c):
    return ((x-c+.5)%1)-.5


class Solver:
    def __init__(self,n,steps):
        self.n=n
        self.steps=steps
        self.dt=DT/steps
        self.ix,self.iy=np.meshgrid(np.arange(n,dtype=np.float32),np.arange(n,dtype=np.float32))
        self.x=(self.ix+.5)/n
        self.y=(self.iy+.5)/n
        kx=(2*np.pi*fft.rfftfreq(n,d=1/n)).astype(np.float32)[None,:]
        ky=(2*np.pi*fft.fftfreq(n,d=1/n)).astype(np.float32)[:,None]
        ksq=kx*kx+ky*ky
        self.inverse=np.divide(1.,ksq,out=np.zeros_like(ksq),where=ksq>0)
        self.kx=1j*kx
        self.ky=1j*ky
        self.kx[0,-1]=0
        self.ky[n//2,0]=0
        self.diff_w=np.exp(-self.dt*.00030*ksq)
        self.diff_rho=np.exp(-self.dt*.00008*ksq)

    def source(self,s,t):
        cx=s["center"]+s["drift"]*math.sin(s["frequency"]*t+s["phase"])
        x=periodic_delta(self.x,cx)
        y=periodic_delta(self.y,s["height"])
        return s["power"]*np.exp(-.5*(x*x+y*y)/(s["radius"]**2))

    def initial(self,s):
        w=np.zeros_like(self.x)
        for x,y,r,a in s["vortices"]:
            w+=a*np.exp(-.5*(periodic_delta(self.x,x)**2+periodic_delta(self.y,y)**2)/(r*r))
        w-=w.mean()
        return np.stack((np.zeros_like(w),w))

    @staticmethod
    def remap(z,x,y):
        return cv2.remap(z,x,y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_WRAP)

    def maccormack(self,z,bx,by,fx,fy,idx):
        moved=self.remap(z,bx,by)
        corrected=moved+.5*(z-self.remap(moved,fx,fy))
        a,b,c,d=idx
        lo=np.minimum(np.minimum(z[b,a],z[b,c]),np.minimum(z[d,a],z[d,c]))
        hi=np.maximum(np.maximum(z[b,a],z[b,c]),np.maximum(z[d,a],z[d,c]))
        return np.clip(corrected,lo,hi)

    def frame(self,state,s,t):
        rho,w=state
        for step in range(self.steps):
            w_hat=fft.rfft2(w)
            psi=w_hat*self.inverse
            u=fft.irfft2(psi*self.ky,s=(self.n,self.n))+s["wind"]
            v=-fft.irfft2(psi*self.kx,s=(self.n,self.n))
            dx,dy=self.dt*self.n*u,self.dt*self.n*v
            bx,by=self.ix-dx,self.iy-dy
            fx,fy=self.ix+dx,self.iy+dy
            xx=np.floor(bx).astype(np.int32)
            yy=np.floor(by).astype(np.int32)
            idx=(xx%self.n,yy%self.n,(xx+1)%self.n,(yy+1)%self.n)
            rho=self.maccormack(rho,bx,by,fx,fy,idx)
            w=self.maccormack(w,bx,by,fx,fy,idx)
            rho=rho*math.exp(-.16*self.dt)+self.dt*self.source(s,t+(step+.5)*self.dt)
            grad=.5*self.n*(np.roll(rho,-1,axis=1)-np.roll(rho,1,axis=1))
            w+=self.dt*s["beta"]*grad
            rho=np.maximum(fft.irfft2(fft.rfft2(rho)*self.diff_rho,s=(self.n,self.n)),0)
            w=fft.irfft2(fft.rfft2(w)*self.diff_w,s=(self.n,self.n))
            w-=w.mean()
        return np.stack((rho,w))


def shrink(state):
    return np.stack([cv2.resize(x,(LOW,LOW),interpolation=cv2.INTER_AREA) for x in state])


def feature(state,provisional,source,s):
    return np.stack((state[0],state[1]/WS,provisional[0],provisional[1]/WS,source*DT,
                     np.full_like(source,s["wind"]*4),np.full_like(source,s["beta"]-1.)))


class FourierLayer(nn.Module):
    def __init__(self,width,modes):
        super().__init__()
        self.modes=modes
        self.pos=nn.Parameter(torch.randn(width,width,modes,modes,dtype=torch.cfloat)*(.6/width))
        self.neg=nn.Parameter(torch.randn(width,width,modes,modes,dtype=torch.cfloat)*(.6/width))
        self.local=nn.Conv2d(width,width,1)

    def forward(self,x):
        m=self.modes
        xh=torch.fft.rfft2(x)
        out=xh.new_zeros(xh.shape)
        out[:,:,:m,:m]=torch.einsum("bixy,ioxy->boxy",xh[:,:,:m,:m],self.pos)
        out[:,:,-m:,:m]=torch.einsum("bixy,ioxy->boxy",xh[:,:,-m:,:m],self.neg)
        return F.gelu(torch.fft.irfft2(out,s=x.shape[-2:])+self.local(x))


class Operator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(7,12,1),FourierLayer(12,8),FourierLayer(12,8),nn.Conv2d(12,2,1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self,x):
        correction=.15*torch.tanh(self.net(x))
        support=F.max_pool2d(x[:,2:3].clamp_min(0),7,stride=1,padding=3)
        return correction*(support*9).clamp(max=1)


def hybrid_step(operator,solver,state,s,t):
    pred=solver.frame(state,s,t)
    data=feature(state,pred,solver.source(s,t+DT/2),s)
    with torch.inference_mode():
        residual=operator(torch.from_numpy(data)[None])[0].numpy()
    pred[0]=np.maximum(pred[0]+residual[0],0)
    pred[1]+=WS*residual[1]
    pred[1]-=pred[1].mean()
    return pred


def ref_scene(s):
    solver=Solver(HIGH,4)
    state=solver.initial(s)
    result=[state]
    for i in range(FRAMES):
        state=solver.frame(state,s,i*DT)
        result.append(state)
    return np.array(result)


def train_data():
    path=ROOT/"data/train.npz"
    if path.exists():
        data=np.load(path)
        return data["x"],data["y"]
    solver=Solver(LOW,1)
    xs,ys=[],[]
    for idx,seed in enumerate(TRAIN):
        s=scene(seed)
        ref=ref_scene(s)
        for i in range(0,FRAMES,2):
            state=shrink(ref[i])
            target=shrink(ref[i+1])
            t=i*DT
            provisional=solver.frame(state,s,t)
            x=feature(state,provisional,solver.source(s,t+DT/2),s)
            y=target-provisional
            y[1]/=WS
            xs.append(x)
            ys.append(y)
        if (idx+1)%11==0:
            print(f"Training scenes {idx+1}/{len(TRAIN)}",flush=True)
    x=np.asarray(xs)
    y=np.asarray(ys)
    np.savez(path,x=x,y=y)
    return x,y


def render_state(state):
    rho=state[0]
    return rho if rho.shape==(HIGH,HIGH) else cv2.resize(rho,(HIGH,HIGH),interpolation=cv2.INTER_LINEAR)


def simulate(s,method,operator):
    grids={"classic48":(48,1),"classic48x4":(48,4),"classic64":(64,1),"classic96":(96,1),
           "classic144":(HIGH,1),"hybrid48":(LOW,1),"reference":(HIGH,4)}
    n,substeps=grids[method]
    solver=Solver(n,substeps)
    state=solver.initial(s)
    images=[]
    durations=[]
    for i in range(FRAMES):
        start=perf_counter()
        if method=="hybrid48":
            state=hybrid_step(operator,solver,state,s,i*DT)
        else:
            state=solver.frame(state,s,i*DT)
        density=render_state(state)
        durations.append(perf_counter()-start)
        images.append(density)
    return np.array(images),durations


def rmse(pred,ref):
    return float(np.sqrt(np.mean((pred-ref)**2,dtype=np.float64)))


def grad_rmse(pred,ref):
    error=pred-ref
    return float(np.sqrt(.5*(np.mean(np.diff(error,axis=1)**2,dtype=np.float64)+
                              np.mean(np.diff(error,axis=2)**2,dtype=np.float64))))


def training():
    x,y=train_data()
    x,y=torch.from_numpy(x),torch.from_numpy(y)
    refs=[ref_scene(scene(seed))[1:,0] for seed in VAL]
    torch.manual_seed(32)
    rng=np.random.default_rng(32)
    model=Operator()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,EPOCHS,eta_min=.0002)
    best=float("inf")
    trace=[]
    for epoch in range(EPOCHS):
        model.train()
        losses=[]
        for idx in np.array_split(rng.permutation(len(x)),math.ceil(len(x)/16)):
            model_input,target=x[idx],y[idx]
            optimizer.zero_grad()
            pred=model(model_input)
            weights=1+3*model_input[:,2:3].clamp(0,1)
            residual=pred-target
            loss=(weights*residual.square()).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        row=dict(epoch=epoch+1,loss=float(np.mean(losses)))
        if (epoch+1)%2==0:
            model.eval()
            error=float(np.mean([rmse(simulate(scene(seed),"hybrid48",model)[0],ref) for seed,ref in zip(VAL,refs)]))
            row["val_rmse"]=error
            if error<best:
                best=error
                torch.save(model.state_dict(),ROOT/"results/operator.pt")
        trace.append(row)
        print(json.dumps(row),flush=True)
    (ROOT/"results/training.json").write_text(json.dumps(trace,indent=2)+"\n")
    model.load_state_dict(torch.load(ROOT/"results/operator.pt",map_location="cpu",weights_only=True))
    return model.eval()


def assess(operator):
    scene_metrics=[]
    timings={m:[] for m in NAMES}
    rng=np.random.default_rng(3200)
    for m in NAMES:
        simulate(scene(999),m,operator)
    for j,seed in enumerate(TEST):
        s=scene(seed,gust=(j>=12))
        results={}
        for m in rng.permutation(NAMES):
            result,elapsed=simulate(s,m,operator)
            results[m]=result
            timings[m].extend(elapsed)
        ref=results["reference"]
        row=dict(seed=seed,strong_wind=j>=12,configuration=s,methods={})
        for m,result in results.items():
            row["methods"][m]=dict(rmse=rmse(result,ref),gradient_rmse=grad_rmse(result,ref))
        scene_metrics.append(row)
        if j in (0,12):
            np.savez_compressed(ROOT/"results"/f"demo_{seed}.npz",**results)
        print("Test scene",seed,"done",flush=True)
    summary={}
    for m in NAMES:
        summary[m]=dict(rmse=float(np.mean([r["methods"][m]["rmse"] for r in scene_metrics])),
                        gradient_rmse=float(np.mean([r["methods"][m]["gradient_rmse"] for r in scene_metrics])),
                        median_ms=float(1000*np.median(timings[m])),p95_ms=float(1000*np.quantile(timings[m],.95)),
                        normal_rmse=float(np.mean([r["methods"][m]["rmse"] for r in scene_metrics if not r["strong_wind"]])),
                        strong_wind_rmse=float(np.mean([r["methods"][m]["rmse"] for r in scene_metrics if r["strong_wind"]])))
    hybrid=summary["hybrid48"]
    comparable=[m for m in NAMES if m.startswith("classic") and summary[m]["median_ms"]<=1.15*hybrid["median_ms"]]
    assert comparable
    best=min(comparable,key=lambda m:summary[m]["rmse"])
    draw=rng.integers(len(scene_metrics),size=(5000,len(scene_metrics)))
    differences={}
    for m in (name for name in NAMES if name.startswith("classic")):
        diff=np.array([r["methods"][m]["rmse"]-r["methods"]["hybrid48"]["rmse"] for r in scene_metrics])
        differences[m]=dict(relative=float(diff.mean()/summary[m]["rmse"]),
                            delta=float(diff.mean()),ci95=np.quantile(diff[draw].mean(axis=1),[.025,.975]).tolist())
    speed=summary["reference"]["median_ms"]/hybrid["median_ms"]
    decision=differences[best]
    report=dict(forecast="T3-2026-0032",scenes=len(TEST),frames_per_scene=FRAMES,
                methods=summary,classic_same_cost=best,differences=differences,speed_vs_reference=speed,
                meets_gate=bool(decision["relative"]>=.05 and decision["ci95"][0]>0 and speed>=2))
    (ROOT/"results/scenes.json").write_text(json.dumps(scene_metrics,indent=2)+"\n")
    (ROOT/"results/timing_seconds.json").write_text(json.dumps(timings,indent=2)+"\n")
    (ROOT/"results.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)


def main():
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    assess(training())


if __name__=="__main__":
    main()
