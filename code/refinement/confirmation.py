import json
from pathlib import Path

import cv2
import numpy as np

import development as dev

ROOT=Path(__file__).resolve().parent
TEST=range(6000,6064)
MODEL=("time_linear96_0.5","linear",96,True,.5)
ABLATION="time_linear96_0"
REFERENCE=("reference","reference",dev.HIGH,False,0.)
CLASSICS=(
    tuple((f"maccormack{n}","maccormack",n,False,0.) for n in (48,64,80,96,112,144))+
    tuple((f"euler_{i}{n}",i,n,False,0.) for i in ("linear","cubic") for n in (64,80,96,104,112,128,144))+
    tuple((dev.variant_name(n,i,t,0.),i,n,t,0.) for i in ("linear","cubic")
          for t in (False,True) for n in (80,96,104,112,128,144))
)
PREVIOUS="previous_filter"


def load_weights():
    rows=json.loads((ROOT/"results/weights.json").read_text())
    weights={tuple(r["setup"]):np.array(r["weights"],dtype=np.float32) for r in rows}
    old=json.loads((ROOT.parent/"retry/results/weights.json").read_text())
    return weights,np.array(old["96"],dtype=np.float32)


def advance(scene,entry,weights,old):
    if entry[0]==PREVIOUS:
        return dev.retry.simulate(scene,"learned",96,old,1.)
    return dev.simulate(scene,entry,weights)


def main():
    cv2.setNumThreads(1)
    weights,old=load_weights()
    models=list(CLASSICS)+[MODEL,(PREVIOUS,"learned",96,False,1.),REFERENCE]
    times={m[0]:[] for m in models}
    scenes=[]
    rng=np.random.default_rng(6032)
    for model in models:
        advance(dev.common.scene(997),model,weights,old)
    for j,seed in enumerate(TEST):
        scene=dev.common.scene(seed,gust=j>=32)
        video={}
        for i in rng.permutation(len(models)):
            model=models[i]
            data,latencies=advance(scene,model,weights,old)
            video[model[0]]=data
            times[model[0]].extend(latencies)
        ref=video["reference"]
        records={k:dict(rmse=dev.common.rmse(v,ref),gradient=dev.common.grad_rmse(v,ref))
                 for k,v in video.items()}
        scenes.append(dict(seed=seed,strong_wind=j>=32,metrics=records))
        if j in (0,32):
            np.savez_compressed(ROOT/"results"/f"demo_{seed}.npz",**video)
        if (j+1)%8==0:
            print(f"Test scenes {j+1}/{len(TEST)}",flush=True)
    stats={}
    for key,latencies in times.items():
        errors=np.array([s["metrics"][key]["rmse"] for s in scenes])
        stats[key]=dict(rmse=float(errors.mean()),
                        gradient=float(np.mean([s["metrics"][key]["gradient"] for s in scenes])),
                        normal_rmse=float(errors[:32].mean()),strong_wind_rmse=float(errors[32:].mean()),
                        median_ms=float(1e3*np.median(latencies)),p95_ms=float(1e3*np.quantile(latencies,.95)))
    name=MODEL[0]
    possible=[m[0] for m in CLASSICS if stats[m[0]]["median_ms"]<=1.15*stats[name]["median_ms"]]
    assert ABLATION in possible
    best=min(possible,key=lambda m:stats[m]["rmse"])
    draws=np.column_stack([rng.integers(0,32,size=(5000,32)),rng.integers(32,64,size=(5000,32))])
    differences={}
    for k in [m[0] for m in CLASSICS]+[PREVIOUS]:
        delta=np.array([s["metrics"][k]["rmse"]-s["metrics"][name]["rmse"] for s in scenes])
        differences[k]=dict(relative=float(delta.mean()/stats[k]["rmse"]),delta=float(delta.mean()),
                            ci95=np.quantile(delta[draws].mean(axis=1),[.025,.975]).tolist(),
                            scenes_ahead=int((delta>0).sum()))
    speed=stats["reference"]["median_ms"]/stats[name]["median_ms"]
    d=differences[best]
    outcome=dict(forecast="T3-2026-0032",scenes=len(TEST),frames_per_scene=dev.FRAMES,
                 model=name,ablation=ABLATION,best_same_cost=best,comparable=possible,
                 speed_vs_reference=speed,methods=stats,differences=differences,
                 meets_gate=bool(d["relative"]>=.05 and d["ci95"][0]>0 and speed>=2))
    (ROOT/"results/scenes.json").write_text(json.dumps(scenes,indent=2)+"\n")
    (ROOT/"results/timing_seconds.json").write_text(json.dumps(times,indent=2)+"\n")
    (ROOT/"results.json").write_text(json.dumps(outcome,indent=2)+"\n")
    print(json.dumps({k:outcome[k] for k in ("model","ablation","best_same_cost","speed_vs_reference","meets_gate")}),flush=True)
    for k in dict.fromkeys((name,ABLATION,best,PREVIOUS,"maccormack96","reference")):
        print(k,stats[k],differences.get(k),flush=True)


if __name__=="__main__":
    main()
