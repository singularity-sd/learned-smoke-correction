import json
from pathlib import Path

import cv2
import numpy as np

import confirmation as test
import development as dev

ROOT=Path(__file__).resolve().parent
SUBSET=(6000,6001,6002,6003,6032,6033,6034,6035)


def main():
    cv2.setNumThreads(1)
    summary=json.loads((ROOT/"results.json").read_text())
    results=json.loads((ROOT/"results/scenes.json").read_text())
    timings=json.loads((ROOT/"results/timing_seconds.json").read_text())
    assert {r["seed"] for r in results}==set(test.TEST) and len(results)==64
    assert set(timings)==set(summary["methods"])
    assert all(len(x)==64*dev.FRAMES for x in timings.values())
    for k,m in summary["methods"].items():
        assert np.isclose(m["rmse"],np.mean([r["metrics"][k]["rmse"] for r in results]))
        assert np.isclose(m["median_ms"],1000*np.median(timings[k]))
    best=summary["best_same_cost"]
    entries={m[0]:m for m in test.CLASSICS}
    weights,previous=test.load_weights()
    rows=[]
    for seed in SUBSET:
        scene=dev.common.scene(seed,gust=seed>=6032)
        solver=dev.common.Solver(192,8)
        state=solver.initial(scene)
        ref=[]
        for i in range(dev.FRAMES):
            state=solver.frame(state,scene,i*dev.DT)
            ref.append(cv2.resize(state[0],(dev.HIGH,dev.HIGH),interpolation=cv2.INTER_AREA))
        ref=np.asarray(ref)
        variants={"learned":test.MODEL,"same_cost_classic":entries[best],"original_reference":test.REFERENCE}
        scores={}
        for key,entry in variants.items():
            video,_=test.advance(scene,entry,weights,previous)
            scores[key]=dev.common.rmse(video,ref)
        rows.append(dict(seed=seed,errors=scores))
    averages={k:float(np.mean([r["errors"][k] for r in rows])) for k in rows[0]["errors"]}
    output=dict(scenes_verified=len(results),timings_verified=sum(map(len,timings.values())),
                sensitivity_grid=192,sensitivity_substeps=8,subset=rows,
                means=averages,relative_gain_vs_same_cost=1-averages["learned"]/averages["same_cost_classic"])
    (ROOT/"results/audit.json").write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps({k:v for k,v in output.items() if k!="subset"},indent=2),flush=True)


if __name__=="__main__":
    main()
