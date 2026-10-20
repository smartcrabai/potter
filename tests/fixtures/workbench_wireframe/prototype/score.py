import json, sys
from pngread import read_png
import wire_model2 as m
def run(geom, oracle, png, variant='first', verbose=True):
    w,h,px=read_png(png)
    th=oracle['theme_view_3d']; bgc=oracle['theme_background']['high_gradient'][:3]
    rec,bg=m.build_records(geom,oracle,th['wire'][:3],bgc,variant)
    res=m.antialias(rec,w,h,bg)
    bgd=tuple(m.display(c) for c in bg)
    model={i:tuple(m.display(c) for c in res.get(i,bg)) for i in range(w*h)}
    def is_bg(i,x,y):
        if model[i]!=bgd: return False
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx,ny=x+dx,y+dy
            if 0<=nx<w and 0<=ny<h and model[ny*w+nx]!=bgd: return False
        return True
    interior=[];cube=[];sph=[];
    ink={'cube':[0,0],'sphere':[0,0]}
    for y in range(150,550):
      for x in range(150,650):
        i=y*w+x
        obj='cube' if x<400 else 'sphere'
        ink[obj][0]+=sum(bgd)-sum(model[i]); ink[obj][1]+=sum(bgd)-sum(px[i])
        if is_bg(i,x,y): interior.append(max(abs(a-b) for a,b in zip(px[i],bgd)))
        else:
            d=max(abs(a-b) for a,b in zip(model[i],px[i]))
            (cube if x<400 else sph).append(d)
    def st(e):
        e=sorted(e); n=len(e)
        return dict(n=n, mean=round(sum(e)/n,3), p95=e[int(.95*(n-1))], max=e[-1], over6=sum(1 for v in e if v>6))
    out=dict(interior_max=max(interior), interior_over1=sum(1 for v in interior if v>1), cube=st(cube), sphere=st(sph),
             ink={k:(v[0],v[1],round(100*(v[0]-v[1])/max(v[1],1),2)) for k,v in ink.items()})
    if verbose: print(variant, json.dumps(out))
    return out
if __name__=='__main__':
    geom=json.load(open('geometry.json')); oracle=json.load(open('final.json'))
    for v in ['first','both']:
        run(geom,oracle,'final.png',v)
