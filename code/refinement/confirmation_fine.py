import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from scipy import fft

import confirmation as previous
import development as dev

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results/fine"
SEEDS=range(7000,7064)
SENSITIVITY=(7000,7001,7002,7003,7032,7033,7034,7035)
FIT_KEY=(96,"linear",True)
ABLATION="time_linear96_0"
BIGGER="time_linear112_0"
MODEL="learned"
REFERENCES=("reference144","reference192")
CLASSICS=list(previous.CLASSICS)+[(f"time_linear{n}_0","linear",n,True,0.) for n in (88,100,108,116,120)]


class SeparableStep(dev.MidStep):
    def __init__(self,weights):
        super().__init__(96,"linear",True)
        self.gx=.5*(weights[:,0,None]*self.sx[0]+weights[:,2,None]*self.sx[0]**2)
        self.gy=.5*(weights[:,1,None]*self.sy[:,0]+weights[:,3,None]*self.sy[:,0]**2)

    def frame(self,state,scene,t):
        spec,factors=self.advance_spectrum(state,scene,t,True)
        gain=factors[0]*self.gx[:,None,:]+factors[1]*self.gy[:,:,None]
        spec*=1+np.clip(gain,-.18,.55)
        out=fft.irfft2(spec,s=self.shape)
        out[0]=np.maximum(out[0],0)
        out[1]-=out[1].mean()
        return out


def equivalence(weights):
    initial=dev.MidStep(*FIT_KEY)
    reduced=SeparableStep(weights)
    error=0.
    for seed in (2000,2003):
        scene=dev.common.scene(seed)
        initial.previous=reduced.previous=None
        ref=dev.common.ref_scene(scene)
        for i in range(dev.FRAMES):
            state=dev.retry.resize(ref[i],96)
            a=initial.frame(state,scene,i*dev.DT,weights,.5)
            b=reduced.frame(state,scene,i*dev.DT)
            error=max(error,float(np.max(np.abs(a-b))))
    assert error<1e-5,error
    return error


def rollout(solver,s):
    state=solver.initial(s)
    density=[]
    seconds=[]
    for i in range(dev.FRAMES):
        start=perf_counter()
        state=solver.frame(state,s,i*dev.DT)
        rho=state[0]
        frame=(rho if solver.n==dev.HIGH else cv2.resize(rho,(dev.HIGH,dev.HIGH),
                interpolation=cv2.INTER_AREA if solver.n>dev.HIGH else cv2.INTER_LINEAR))
        seconds.append(perf_counter()-start)
        density.append(frame)
    result=np.array(density)
    assert np.isfinite(result).all()
    return result,seconds


def solver_for(key,entries,weights):
    if key==MODEL:
        return SeparableStep(weights)
    if key=="reference144":
        return dev.common.Solver(144,4)
    if key=="reference192":
        return dev.common.Solver(192,8)
    name,kind,n,temporal,_=entries[key]
    if kind=="maccormack":
        return dev.common.Solver(n,1)
    if name.startswith(("mid_","time_")):
        return dev.MidStep(n,kind,temporal)
    return dev.retry.FastStep(n,cv2.INTER_CUBIC if kind=="cubic" else cv2.INTER_LINEAR)


def summarize(rows,times):
    rng=np.random.default_rng(7064)
    draw=np.column_stack((rng.integers(0,32,size=(10000,32)),rng.integers(32,64,size=(10000,32))))
    timing={k:dict(median_ms=float(np.median(v)*1000),p95_ms=float(np.quantile(v,.95)*1000)) for k,v in times.items()}
    comparables=[k for k in times if k not in (*REFERENCES,MODEL) and
                 timing[k]["median_ms"]<=1.15*timing[MODEL]["median_ms"]]
    strict=[k for k in comparables if timing[k]["median_ms"]<=timing[MODEL]["median_ms"]]
    comparisons={}
    for ref in REFERENCES:
        matrix={k:np.array([r["error"][ref][k]["rmse"] for r in rows]) for k in times}
        stats={k:dict(mean=float(x.mean()),normal=float(x[:32].mean()),gust=float(x[32:].mean()),
                      gradient=float(np.mean([r["error"][ref][k]["gradient"] for r in rows])))
               for k,x in matrix.items()}
        best=min(comparables,key=lambda k:stats[k]["mean"])
        strict_best=min(strict,key=lambda k:stats[k]["mean"])
        diff={}
        for k in times:
            if k in (*REFERENCES,MODEL):
                continue
            delta=matrix[k]-matrix[MODEL]
            ratio=matrix[MODEL][draw].mean(axis=1)/matrix[k][draw].mean(axis=1)-1
            diff[k]=dict(relative_gain=float(delta.mean()/stats[k]["mean"]),
                         ci95_absolute=np.quantile(delta[draw].mean(axis=1),[.025,.975]).tolist(),
                         relative_error_increase_ci95=np.quantile(ratio,[.025,.975]).tolist(),
                         wins=int((delta>0).sum()))
        comparisons[ref]=dict(rmse=stats,comparables=comparables,best=best,best_strict_cost=strict_best,differences=diff)
    fine=comparisons["reference192"]
    best=fine["best"]
    delta=fine["differences"][best]
    speed=timing["reference192"]["median_ms"]/timing[MODEL]["median_ms"]
    saving=1-timing[MODEL]["median_ms"]/timing[BIGGER]["median_ms"]
    return dict(forecast="T3-2026-0032",scenes=len(rows),frames_per_scene=dev.FRAMES,
                timing=timing,comparisons=comparisons,speed_vs_fine=speed,
                savings_vs_112=saving,meets_quality_gate=bool(delta["relative_gain"]>=.05 and
                    delta["ci95_absolute"][0]>0 and speed>=2),
                meets_latency_gate=bool(saving>=.10 and
                    fine["differences"][BIGGER]["relative_error_increase_ci95"][1]<=.05))


def main():
    cv2.setNumThreads(1)
    all_weights,_=previous.load_weights()
    weights=all_weights[FIT_KEY]
    error=equivalence(weights)
    print("Fused step max difference",error,flush=True)
    entries={e[0]:e for e in CLASSICS}
    keys=list(entries)+[MODEL,*REFERENCES]
    times={k:[] for k in keys}
    rows=[]
    rng=np.random.default_rng(7032)
    for k in keys:
        rollout(solver_for(k,entries,weights),dev.common.scene(998))
    for i,seed in enumerate(SEEDS):
        scene=dev.common.scene(seed,gust=i>=32)
        video={}
        for k in rng.permutation(keys):
            video[k],latency=rollout(solver_for(k,entries,weights),scene)
            times[k].extend(latency)
        scores={ref:{k:dict(rmse=dev.common.rmse(v,video[ref]),gradient=dev.common.grad_rmse(v,video[ref]))
                     for k,v in video.items()} for ref in REFERENCES}
        rows.append(dict(seed=seed,gust=i>=32,error=scores))
        if i in (0,32):
            selected=("reference144","reference192",MODEL,ABLATION,"time_linear104_0",BIGGER,"time_linear108_0")
            np.savez_compressed(OUT/f"demo_{seed}.npz",**{k:video[k] for k in selected})
        if (i+1)%8==0:
            print(f"Fine-reference test {i+1}/{len(SEEDS)}",flush=True)
    report=summarize(rows,times)
    report["fused_step_max_absolute_difference"]=error
    (OUT/"scenes.json").write_text(json.dumps(rows,indent=2))
    (OUT/"timing_seconds.json").write_text(json.dumps(times,indent=2))
    (ROOT/"results_fine.json").write_text(json.dumps(report,indent=2)+"\n")
    for ref,c in report["comparisons"].items():
        print(ref,"best",c["best"],"strict cost",c["best_strict_cost"],flush=True)
        for key in dict.fromkeys((MODEL,c["best"],c["best_strict_cost"],ABLATION,BIGGER)):
            print(key,c["rmse"][key],report["timing"][key],c["differences"].get(key),flush=True)
    print({k:report[k] for k in ("meets_quality_gate","meets_latency_gate","speed_vs_fine","savings_vs_112")},flush=True)
    detail=[]
    chosen=report["comparisons"]["reference192"]["best"]
    for seed in SENSITIVITY:
        s=dev.common.scene(seed,gust=seed>=7032)
        truth,_=rollout(dev.common.Solver(256,16),s)
        values={}
        for k in dict.fromkeys((MODEL,chosen,BIGGER,"reference192")):
            v,_=rollout(solver_for(k,entries,weights),s)
            values[k]=dev.common.rmse(v,truth)
        detail.append(dict(seed=seed,rmse=values))
    means={k:float(np.mean([row["rmse"][k] for row in detail])) for k in detail[0]["rmse"]}
    (OUT/"sensitivity256.json").write_text(json.dumps(dict(scenes=detail,means=means),indent=2)+"\n")
    print("Sensitivity against 256x256 with 16 substeps",means,flush=True)


if __name__=="__main__":
    main()
