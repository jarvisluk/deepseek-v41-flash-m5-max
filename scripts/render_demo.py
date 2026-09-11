"""Render a sped-up instrumented replay from real inference JSONL events.

The timeline is reconstructed from measured token/layer timestamps, not from
simulated generation. Cumulative logical checkpoint reads include OS cache hits.
"""
import argparse, bisect, json, math, re, subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

W,H,FPS=1280,800,12
BG='#0b1118'; PANEL='#121d28'; BORDER='#263444'; WHITE='#edf4fa'
MUTED='#91a4b8'; CYAN='#6ee7da'; GOLD='#f0c56a'; BLUE='#8caeff'
FONT='/System/Library/Fonts/Supplemental/Arial.ttf'
BOLD='/System/Library/Fonts/Supplemental/Arial Bold.ttf'
MONO='/System/Library/Fonts/SFNSMono.ttf'
def font(size,bold=False,mono=False): return ImageFont.truetype(MONO if mono else BOLD if bold else FONT,size)

def load_events(path):
    layers=[]; tokens=[]; summary=None; offset=0.; prompt=''
    for line in Path(path).read_text().splitlines():
        try: v=json.loads(line)
        except ValueError: continue
        if 'input_ids' in v: prompt=v['prompt']
        if 'layer' in v:
            layers.append(dict(v,t=offset+v['seconds']))
        if 'step' in v:
            tokens.append(dict(v,t=v['seconds']))
            offset=v['seconds']
        if v.get('event')=='complete':
            summary=v
            layers.append(dict(layer=39,t=v['total_seconds'],read_gib=v['read_gib'],peak_gib=v['peak_mlx_gib']))
    if not tokens: raise ValueError('No measured tokens in log')
    return layers,tokens,summary

def wrap(draw,text,ft,width):
    lines=[]
    for para in text.split('\n'):
        words=para.split(' '); line=''
        for word in words:
            candidate=(line+' '+word).strip()
            if draw.textlength(candidate,font=ft)>width and line:
                lines.append(line); line=word
            else: line=candidate
        lines.append(line)
    return lines

def mmss(seconds):
    s=int(seconds); return f'{s//60:02d}:{s%60:02d}'

def frame(t,layers,tokens,total,speed,finished=False,optimized=False):
    im=Image.new('RGB',(W,H),BG); d=ImageDraw.Draw(im)
    def text(x,y,s,size=20,color=WHITE,bold=False,mono=False):
        d.text((x,y),s,font=font(size,bold,mono),fill=color)
    def box(x,y,w,h,fill=PANEL,r=15):
        d.rounded_rectangle((x,y,x+w,y+h),radius=r,fill=fill,outline=BORDER,width=1)
    li=bisect.bisect_right([v['t'] for v in layers],t)-1
    ti=bisect.bisect_right([v['t'] for v in tokens],t)-1
    layer=layers[li] if li>=0 else dict(layer=-1,read_gib=0,peak_gib=0,t=0)
    last=tokens[ti] if ti>=0 else None
    done=finished or t>=total
    d.rectangle((0,0,W,5),fill=CYAN)
    text(40,29,'OPTIMIZED LOCAL INFERENCE / MEASURED TOKEN REPLAY' if optimized else 'LOCAL INFERENCE / MEASURED TOKEN REPLAY',14,CYAN,True)
    text(40,56,'Mac mini M1 · 16 GB RAM · 1 TB SSD',36,bold=True)
    text(40,104,'DeepSeek V4.1 Flash / Original FP4–FP8 checkpoint',21,MUTED)
    box(1040,29,200,42,fill='#16312f',r=21)
    text(1066,40,f'{speed:g}× PLAYBACK',18,CYAN,True)
    text(1043,87,'8-core CPU · 8-core GPU',15,MUTED)
    text(1043,108,'Original FP4 / FP8 weights',15,MUTED)
    # Each counter refers to real wall-clock time, independent of playback speed.
    metrics=[('REAL ELAPSED',mmss(min(t,total)),'inference time · minutes : seconds'),
             ('TTFT / FIRST TOKEN',f'{tokens[0]["t"]:.1f} s' if last else 'Waiting…','inference start → first token'),
             ('DECODE SPEED',f'{(last["t"]-tokens[0]["t"])/ti:.1f} s/token' if ti>0 else '—','average after first token')]
    for i,(label,value,note) in enumerate(metrics):
        x=40+i*404; box(x,151,388,115)
        text(x+21,168,label,13,MUTED,True)
        text(x+20,190,value,32,CYAN if i==0 else WHITE,True,mono=(i==0))
        text(x+21,236,note,14,MUTED)
    box(40,285,1200,68)
    text(61,300,'PROMPT',12,CYAN,True)
    text(61,320,'explain what is navier stokes',23,WHITE)
    box(40,372,760,285)
    text(62,391,'MODEL OUTPUT',13,MUTED,True)
    text(590,391,f'{ti+1:02d} / {len(tokens):02d} tokens',15,CYAN,mono=True)
    content=last['text'] if last else ''
    # Render Markdown emphasis as plain typography without changing the words.
    content=content.replace('**','').replace('### ','').replace('## ','')
    if not content:
        text(63,446,'Reading the prompt and model layers…',24,MUTED)
    else:
        ft=font(28); lines=wrap(d,content,ft,710)
        for j,line in enumerate(lines[-5:]):
            d.text((63,435+j*39),line,font=ft,fill=WHITE)
        if not done and int(t*FPS/speed)%12<8:
            j=min(len(lines),5)-1
            end=d.textlength(lines[-1],font=ft)
            d.rectangle((63+end+5,440+j*39,66+end+5,462+j*39),fill=CYAN)
    # Streaming visualization is schematic; counters are actual reader telemetry.
    box(820,372,420,285)
    text(842,391,'SSD → MEMORY / ON-DEMAND WEIGHTS',13,MUTED,True)
    box(842,424,153,64,fill='#0d1722',r=9)
    text(858,434,'CHECKPOINT',11,MUTED,True)
    text(858,452,'475 GiB',24,WHITE,True)
    box(1060,424,157,64,fill='#10312e',r=9)
    text(1076,434,'MLX PEAK',11,CYAN,True)
    text(1076,452,(f'{layer["peak_gib"]:.2f} GiB' if li>=0 else '—'),24,WHITE,True)
    d.line((1008,457,1045,457),fill=CYAN,width=2)
    d.polygon([(1045,457),(1036,452),(1036,462)],fill=CYAN)
    if not done:
        dot=1009+int((t%3)/3*32)
        d.ellipse((dot-3,454,dot+3,460),fill=WHITE)
    text(842,510,'CHECKPOINT BYTES REQUESTED',12,MUTED,True)
    text(842,533,(f'{layer["read_gib"]:,.1f} GiB' if li>=0 else '—'),31,GOLD,True)
    text(842,578,'Cumulative logical reads; includes OS cache.',14,MUTED)
    text(842,601,'4 GiB weight cache + compiled decoding.' if optimized else 'Dense layers + selected experts + lookup rows.',14,MUTED)
    text(842,624,'MLX peak is not whole-system RAM usage.',13,MUTED)
    # A layer-completion trace, not a fabricated hardware utilization measure.
    phase='EXCERPT COMPLETE' if done else ('PREFILL' if last is None else 'GENERATING')
    text(40,682,phase,13,CYAN,True)
    layer_n=40 if done else layer['layer']+1
    for i in range(40):
        x=205+i*20
        d.rounded_rectangle((x,683,x+14,695),radius=2,fill=CYAN if i<layer_n else BORDER)
    text(1030,679,f'Layer {layer_n:02d} / 40',17,MUTED,mono=True)
    if done:
        text(40,723,f'{len(tokens)}-token excerpt · generation stopped at the capture limit.',17,WHITE)
    else:
        wait=t-(last['t'] if last else 0)
        text(40,723,f'Current token wait: {wait:.1f} real seconds',17,WHITE)
    text(40,756,'Recorded from a real local run · custom native SSD-streaming MLX runner · no additional requantization',14,MUTED)
    return im

def main():
    p=argparse.ArgumentParser(); p.add_argument('log'); p.add_argument('--out',default='navier-stokes-demo.mp4'); p.add_argument('--speed',type=float,default=32); p.add_argument('--preview',action='store_true'); p.add_argument('--optimized',action='store_true'); a=p.parse_args()
    layers,tokens,summary=load_events(a.log)
    total=max(summary['total_seconds'],tokens[-1]['t']) if summary else tokens[-1]['t']
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    if a.preview:
        frame(total,layers,tokens,total if summary else total+1,a.speed,bool(summary),a.optimized).save(out.with_suffix('.png')); return
    if not summary: raise ValueError('Run incomplete; refusing to render a final demo')
    duration=total/a.speed+4
    cmd=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','medium','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(out)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    for i in range(math.ceil(duration*FPS)):
        t=min(i/FPS*a.speed,total)
        proc.stdin.write(frame(t,layers,tokens,total,a.speed,optimized=a.optimized).tobytes())
    proc.stdin.close()
    if proc.wait(): raise RuntimeError('ffmpeg failed')
    frame(total,layers,tokens,total,a.speed,True,a.optimized).save(out.with_suffix('.png'))
    print(json.dumps({'file':str(out),'seconds':duration,'speed':a.speed,'real_seconds':total,'metrics':summary},indent=2))

if __name__=='__main__':main()
