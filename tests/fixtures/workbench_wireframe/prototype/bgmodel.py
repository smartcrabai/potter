import math
def s2l(c):
    return c / 12.92 if c < 0.04045 else ((c + 0.055) / 1.055) ** 2.4
def l2s(c):
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055
def g22(c):
    return c ** (1 / 2.2)
MEAS = {0: 0, 0.25: 66, 0.5: 127, 0.75: 189, 1.0: 255, 0.2392156862745098: 63}
cands = {
 'srgb-float->srgb-enc, round': lambda x: round(l2s(s2l(x)) * 255),
 'srgb-float->srgb-enc, floor': lambda x: math.floor(l2s(s2l(x)) * 255),
 'srgb->lin, UNORM8 lin, srgbenc round': lambda x: round(l2s(round(s2l(x)*255)/255) * 255),
 'srgb->lin, UNORM8 lin, srgbenc floor': lambda x: math.floor(l2s(round(s2l(x)*255)/255) * 255),
 'srgb->lin, UNORM8, gamma2.2 round': lambda x: round(g22(round(s2l(x)*255)/255) * 255),
 'srgb->lin, UNORM8, gamma2.2 floor': lambda x: math.floor(g22(round(s2l(x)*255)/255) * 255),
 'srgb->lin, gamma2.2 round': lambda x: round(g22(s2l(x)) * 255),
 'srgb->lin, gamma2.2 floor': lambda x: math.floor(g22(s2l(x)) * 255),
 'raw x round': lambda x: round(x*255),
}
for name, f in cands.items():
    errs = [(k, f(k), v) for k, v in MEAS.items()]
    print(f"{name:40s}", [ (round(k,4), got, want) for k,got,want in errs], 'maxerr', max(abs(g-w) for _,g,w in errs))
