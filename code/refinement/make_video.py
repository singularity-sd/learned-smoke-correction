import json
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
import crea_demo as visual


def label(key):
    return key.replace("time_linear","TEMPORAL LINEAR ").replace("time_cubic","TEMPORAL CUBIC ").replace(
        "mid_linear","MIDPOINT LINEAR ").replace("mid_cubic","MIDPOINT CUBIC ").replace(
        "maccormack","MACCORMACK ").replace("euler_linear","EULER LINEAR ").replace(
        "euler_cubic","EULER CUBIC ").removesuffix("_0")


def main():
    report=json.loads((ROOT/"results.json").read_text())
    classic,model=report["best_same_cost"],report["model"]
    visual.LABELS[classic]=label(classic)
    visual.LABELS[model]="LEARNED SPECTRAL / 96 x 96"
    ffmpeg=shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required")
    cv2.setNumThreads(1)
    args=[ffmpeg,"-hide_banner","-loglevel","error","-y","-f","rawvideo","-pix_fmt","bgr24",
          "-s",f"{visual.WIDTH}x{visual.HEIGHT}","-r","16","-i","-","-an","-c:v","libx264",
          "-pix_fmt","yuv420p","-movflags","+faststart",str(ROOT/"smoke_comparison.mp4")]
    process=subprocess.Popen(args,stdin=subprocess.PIPE)
    with process.stdin as video:
        for seed in (6000,6032):
            with np.load(ROOT/"results"/f"demo_{seed}.npz") as data:
                scale=float(np.quantile(data["reference"],.995))
                for t in range(len(data["reference"])):
                    image=visual.frame(data,("reference",classic,model),seed,scale,t)
                    video.write(image.tobytes())
                    if seed==6000 and t==31:
                        assert cv2.imwrite(str(ROOT/"fig_comparison.png"),image)
    if process.wait()!=0:
        raise RuntimeError("ffmpeg failed")


if __name__=="__main__":
    main()
