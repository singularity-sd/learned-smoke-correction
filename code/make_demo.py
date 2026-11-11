import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
LABELS={
    "classic48":"CLASSIC 48 x 48",
    "classic48x4":"CLASSIC 48 x 48 / 4 steps",
    "classic64":"CLASSIC 64 x 64",
    "classic96":"CLASSIC 96 x 96",
    "classic144":"CLASSIC 144 x 144 / 1 step",
    "reference":"REFERENCE 144 x 144 / 4 steps",
    "hybrid48":"FOURIER OPERATOR / 48 x 48",
}
TILE=352
GAP=12
TITLE=52
FOOTER=28
WIDTH=3*TILE+4*GAP
HEIGHT=TITLE+TILE+FOOTER


def write(canvas,text,xy,scale=.52,color=(238,241,246)):
    cv2.putText(canvas,text,xy,cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)


def frame(data,methods,seed,scale,t):
    output=np.full((HEIGHT,WIDTH,3),(24,19,17),dtype=np.uint8)
    write(output,f"Scene {seed}   /   Frame {t+1:02d} of {len(data['reference'])}",
          (GAP,21),.58)
    for index,name in enumerate(methods):
        x=GAP+(TILE+GAP)*index
        write(output,LABELS[name],(x,45),.47,(188,215,232))
        raw=data[name][t][::-1]
        smoke=np.sqrt((raw/scale).clip(0,1))
        shade=np.array([233,218,196],dtype=np.float32)
        shadow=np.array([36,27,21],dtype=np.float32)
        raster=(shadow+(shade-shadow)*smoke[:,:,None]).astype(np.uint8)
        output[TITLE:TITLE+TILE,x:x+TILE]=cv2.resize(raster,(TILE,TILE),interpolation=cv2.INTER_LINEAR)
    write(output,"Same density scale within each scene. Vertical direction is upward.",(GAP,HEIGHT-8),.44)
    return output


def main():
    cv2.setNumThreads(1)
    result=json.loads((ROOT/"results.json").read_text())
    methods=("reference",result["classic_same_cost"],"hybrid48")
    ffmpeg=shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to encode the demo")
    cmd=[ffmpeg,"-hide_banner","-loglevel","error","-y","-f","rawvideo","-pix_fmt","bgr24",
         "-s",f"{WIDTH}x{HEIGHT}","-r","16","-i","-","-an","-c:v","libx264","-pix_fmt","yuv420p",
         "-movflags","+faststart",str(ROOT/"smoke_comparison.mp4")]
    child=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    with child.stdin as video:
        for seed in (3000,3012):
            with np.load(ROOT/"results"/f"demo_{seed}.npz") as data:
                scale=float(np.quantile(data["reference"],.995))
                for t in range(len(data["reference"])):
                    canvas=frame(data,methods,seed,scale,t)
                    video.write(canvas.tobytes())
                    if seed==3000 and t==31:
                        assert cv2.imwrite(str(ROOT/"fig_comparison.png"),canvas)
    if child.wait()!=0:
        raise RuntimeError("ffmpeg encoding failed")


if __name__=="__main__":
    main()
