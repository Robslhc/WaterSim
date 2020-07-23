import taichi as ti

FLUID = 0
AIR = 1
SOLID = 2


# util functions
@ti.pyfunc
def clamp(x, a, b):
    return max(a, min(b, x))


@ti.pyfunc
def vec2(x, y):
    return ti.Vector([x, y])


@ti.pyfunc
def vec3(x, y, z):
    return ti.Vector([x, y, z])


# color map
@ti.data_oriented
class ColorMap:
    # reference: https://gitee.com/citadel2020/taichi_demos/blob/master/mgpcgflip/mgpcgflip.py

    def __init__(self, h, wl, wr, c):
        self.h = h
        self.wl = wl
        self.wr = wr
        self.c = c

    def clamp(self, x):
        return clamp(x, 0.0, 1.0)

    def map(self, x):
        w = 0.0
        if x < self.c:
            w = self.wl
        else:
            w = self.wr
        return self.clamp((w - abs(self.clamp(x) - self.c)) / w * self.h)
