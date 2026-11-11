import json
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import numpy as np

import confirmation_fine as fine

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
import crea_demo as layout


def main():
    result=json.loads((ROOT/"results_fine.json").read_text())
    best=result["comparisons"]["reference192"]["best"]
    layout.LABELS["reference"]="REFERENCE 192 x 192 / 8 steps"
    layout.LABELS["learned"]="LEARNED SPECTRAL / 96 x 96"
    layout.LABELS[best]=best.removesuffix("_0").replace("_"," ").upper()
    executable=shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError("ffmpeg is required")
    cv2.setNumThreads(1)
    weights,_=fine.previous.load_weights()
    entries={entry[0]:entry for entry in fine.CLASSICS}
    args=[executable,"-hide_banner","-loglevel","error","-y","-f","rawvideo","-pix_fmt","bgr24",
          "-s",f"{layout.WIDTH}x{layout.HEIGHT}","-r","16","-i","-","-an","-c:v","libx264","-pix_fmt","yuv420p",
          "-movflags","+faststart",str(ROOT/"smoke_comparison_fine.mp4")]
    child=subprocess.Popen(args,stdin=subprocess.PIPE)
    with child.stdin as video:
        for seed in (7000,7032):
            with np.load(fine.OUT/f"demo_{seed}.npz") as stored:
                data={key:stored[key] for key in stored.files}
            data["reference"]=data["reference192"]
            if best not in data:
                s=fine.dev.common.scene(seed,gust=seed>=7032)
                data[best]=fine.rollout(fine.solver_for(best,entries,weights[fine.FIT_KEY]),s)[0]
            scale=float(np.quantile(data["reference"],.995))
            for frame in range(fine.dev.FRAMES):
                canvas=layout.frame(data,("reference",best,"learned"),seed,scale,frame)
                video.write(canvas.tobytes())
                if seed==7000 and frame==31:
                    assert cv2.imwrite(str(ROOT.parent.parent/"fig_frame.png"),canvas)
    if child.wait()!=0:
        raise RuntimeError("ffmpeg encoding failed")


if __name__=="__main__":
    main()
