import taichi as ti
from CGSolver import CGSolver
from MICPCGSolver import MICPCGSolver
from MGPCGSolver import MGPCGSolver
import numpy as np
from utils import ColorMap, vec2, vec3, clamp
import utils
import random
import time

ti.init(arch=ti.gpu, default_fp=ti.f32)

# params in simulation
cell_res = 256
npar = 2

m = cell_res
n = cell_res
w = 10
h = 10 * n / m
grid_x = w / m
grid_y = h / n
pspace_x = grid_x / npar
pspace_y = grid_y / npar

rho = 1000
g = -9.8
substeps = 4

# algorithm = 'FLIP/PIC'
# algorithm = 'Euler'
algorithm = 'APIC'
FLIP_blending = 0.0

# params in render
screen_res = (400, 400 * n // m)
bwrR = ColorMap(1.0, .25, 1, .5)
bwrG = ColorMap(1.0, .5, .5, .5)
bwrB = ColorMap(1.0, 1, .25, .5)
color_buffer = ti.Vector.field(3, dtype=ti.f32, shape=screen_res)
gui = ti.GUI("watersim2D", screen_res)

# cell type
cell_type = ti.field(dtype=ti.i32, shape=(m, n))

# velocity field
u = ti.field(dtype=ti.f32, shape=(m + 1, n))
v = ti.field(dtype=ti.f32, shape=(m, n + 1))
u_temp = ti.field(dtype=ti.f32, shape=(m + 1, n))
v_temp = ti.field(dtype=ti.f32, shape=(m, n + 1))
u_last = ti.field(dtype=ti.f32, shape=(m + 1, n))
v_last = ti.field(dtype=ti.f32, shape=(m, n + 1))
u_weight = ti.field(dtype=ti.f32, shape=(m + 1, n))
v_weight = ti.field(dtype=ti.f32, shape=(m, n + 1))

# pressure field
p = ti.field(dtype=ti.f32, shape=(m, n))

#pressure solver
preconditioning = 'MG'

MIC_blending = 0.97

mg_level = 4
pre_and_post_smoothing = 2
bottom_smoothing = 10

if preconditioning == None:
    solver = CGSolver(m, n, u, v, cell_type)
elif preconditioning == 'MIC':
    solver = MICPCGSolver(m, n, u, v, cell_type, MIC_blending=MIC_blending)
elif preconditioning == 'MG':
    solver = MGPCGSolver(m,
                         n,
                         u,
                         v,
                         cell_type,
                         multigrid_level=mg_level,
                         pre_and_post_smoothing=pre_and_post_smoothing,
                         bottom_smoothing=bottom_smoothing)

# particle x and v
particle_positions = ti.Vector.field(2, dtype=ti.f32, shape=(m, n, npar, npar))
particle_velocities = ti.Vector.field(2,
                                      dtype=ti.f32,
                                      shape=(m, n, npar, npar))
# particle C
cp_x = ti.Vector.field(2, dtype=ti.f32, shape=(m, n, npar, npar))
cp_y = ti.Vector.field(2, dtype=ti.f32, shape=(m, n, npar, npar))

# particle type
particle_type = ti.field(dtype=ti.f32, shape=(m, n, npar, npar))
P_FLUID = 1
P_OTHER = 0

# extrap utils
valid = ti.field(dtype=ti.i32, shape=(m + 1, n + 1))
valid_temp = ti.field(dtype=ti.i32, shape=(m + 1, n + 1))

# save to gif
result_dir = "./results"
video_manager = ti.VideoManager(output_dir=result_dir,
                                framerate=24,
                                automatic_build=False)


def render():
    @ti.func
    def map_color(c):
        return vec3(bwrR.map(c), bwrG.map(c), bwrB.map(c))

    @ti.kernel
    def fill_marker():
        for i, j in color_buffer:
            x = int((i + 0.5) / screen_res[0] * w / grid_x)
            y = int((j + 0.5) / screen_res[1] * h / grid_y)

            m = cell_type[x, y]

            color_buffer[i, j] = map_color(m * 0.5)

    fill_marker()
    img = color_buffer.to_numpy()
    gui.set_image(img)
    # gui.show()

    # save video
    video_manager.write_frame(img)


def init():
    # init scene
    @ti.kernel
    def init_dambreak(x: ti.f32, y: ti.f32):
        xn = int(x / grid_x)
        yn = int(y / grid_y)

        for i, j in cell_type:
            if i == 0 or i == m - 1 or j == 0 or j == n - 1:
                cell_type[i, j] = utils.SOLID  # boundary
            else:
                if i <= xn and j <= yn:
                    cell_type[i, j] = utils.FLUID
                else:
                    cell_type[i, j] = utils.AIR

    #init simulation
    @ti.kernel
    def init_field():
        for i, j in u:
            u[i, j] = 0.0
            u_last[i, j] = 0.0

        for i, j in v:
            v[i, j] = 0.0
            v_last[i, j] = 0.0

        for i, j in p:
            p[i, j] = 0.0

    @ti.kernel
    def init_particles():
        for i, j, ix, jx in particle_positions:
            if cell_type[i, j] == utils.FLUID:
                particle_type[i, j, ix, jx] = P_FLUID
            else:
                particle_type[i, j, ix, jx] = 0

            px = i * grid_x + (ix + random.random()) * pspace_x
            py = j * grid_y + (jx + random.random()) * pspace_y

            particle_positions[i, j, ix, jx] = vec2(px, py)
            particle_velocities[i, j, ix, jx] = vec2(0.0, 0.0)
            cp_x[i, j, ix, jx] = vec2(0.0, 0.0)
            cp_y[i, j, ix, jx] = vec2(0.0, 0.0)

    init_dambreak(4, 4)
    init_field()
    init_particles()


# -------------- Helper Functions -------------------
@ti.func
def is_valid(i, j):
    return i >= 0 and i < m and j >= 0 and j < n


@ti.func
def is_fluid(i, j):
    return is_valid(i, j) and cell_type[i, j] == utils.FLUID


@ti.func
def is_solid(i, j):
    return is_valid(i, j) and cell_type[i, j] == utils.SOLID


@ti.func
def is_air(i, j):
    return is_valid(i, j) and cell_type[i, j] == utils.AIR


@ti.func
def pos_to_stagger_idx(pos, stagger):
    pos[0] = clamp(pos[0], stagger[0] * grid_x,
                   w - 1e-4 - grid_x + stagger[0] * grid_x)
    pos[1] = clamp(pos[1], stagger[1] * grid_y,
                   h - 1e-4 - grid_y + stagger[1] * grid_y)
    p_grid = pos / vec2(grid_x, grid_y) - stagger
    I = ti.cast(ti.floor(p_grid), ti.i32)

    return I, p_grid


@ti.func
def sample_bilinear(x, source_pos, stagger):
    I, p_grid = pos_to_stagger_idx(source_pos, stagger)
    f = p_grid - I
    g = 1 - f

    return x[I] * (g[0] * g[1]) + x[I + vec2(1, 0)] * (f[0] * g[1]) + x[
        I + vec2(0, 1)] * (g[0] * f[1]) + x[I + vec2(1, 1)] * (f[0] * f[1])


@ti.func
def sample_velocity(pos, u, v):
    u_p = sample_bilinear(u, pos, vec2(0, 0.5))
    v_p = sample_bilinear(v, pos, vec2(0.5, 0))

    return vec2(u_p, v_p)


# -------------- Simulation Steps -------------------
@ti.kernel
def apply_gravity(dt: ti.f32):
    for i, j in v:
        v[i, j] += g * dt


@ti.kernel
def enforce_boundary():
    # u solid
    for i, j in u:
        if is_solid(i - 1, j) or is_solid(i, j):
            u[i, j] = 0.0

    # v solid
    for i, j in v:
        if is_solid(i, j - 1) or is_solid(i, j):
            v[i, j] = 0.0


def extrapolate_velocity():
    # reference: https://gitee.com/citadel2020/taichi_demos/blob/master/mgpcgflip/mgpcgflip.py
    @ti.kernel
    def mark_valid_u():
        for i, j in u:
            # NOTE that the the air-liquid interface is valid
            if is_fluid(i - 1, j) or is_fluid(i, j):
                valid[i, j] = 1
            else:
                valid[i, j] = 0

    @ti.kernel
    def mark_valid_v():
        for i, j in v:
            # NOTE that the the air-liquid interface is valid
            if is_fluid(i, j - 1) or is_fluid(i, j):
                valid[i, j] = 1
            else:
                valid[i, j] = 0

    @ti.kernel
    def diffuse_quantity(dst: ti.template(), src: ti.template(),
                         valid_dst: ti.template(), valid: ti.template()):
        for i, j in dst:
            if valid[i, j] == 0:
                s = 0.0
                count = 0
                for m, n in ti.static(ti.ndrange((-1, 2), (-1, 2))):
                    if 1 == valid[i + m, j + n]:
                        s += src[i + m, j + n]
                        count += 1
                if count > 0:
                    dst[i, j] = s / float(count)
                    valid_dst[i, j] = 1

    mark_valid_u()
    for i in range(10):
        u_temp.copy_from(u)
        valid_temp.copy_from(valid)
        diffuse_quantity(u, u_temp, valid, valid_temp)

    mark_valid_v()
    for i in range(10):
        v_temp.copy_from(v)
        valid_temp.copy_from(valid)
        diffuse_quantity(v, v_temp, valid, valid_temp)


def solve_pressure(dt):
    scale_A = dt / (rho * grid_x * grid_x)
    scale_b = 1 / grid_x

    solver.system_init(scale_A, scale_b)
    solver.solve(500)

    p.copy_from(solver.p)


@ti.kernel
def apply_pressure(dt: ti.f32):
    scale = dt / (rho * grid_x)

    for i, j in ti.ndrange(m, n):
        if is_fluid(i - 1, j) or is_fluid(i, j):
            if is_solid(i - 1, j) or is_solid(i, j):
                u[i, j] = 0
            else:
                u[i, j] -= scale * (p[i, j] - p[i - 1, j])

        if is_fluid(i, j - 1) or is_fluid(i, j):
            if is_solid(i, j - 1) or is_solid(i, j):
                v[i, j] = 0
            else:
                v[i, j] -= scale * (p[i, j] - p[i, j - 1])


@ti.kernel
def update_particle_velocities(dt: ti.f32):
    for p in ti.grouped(particle_positions):
        if particle_type[p] == P_FLUID:
            pv = sample_velocity(particle_positions[p], u, v)
            particle_velocities[p] = pv


@ti.kernel
def advect_particles(dt: ti.f32):
    for p in ti.grouped(particle_positions):
        if particle_type[p] == P_FLUID:
            pos = particle_positions[p]
            pv = particle_velocities[p]

            pos += pv * dt

            particle_positions[p] = pos


@ti.kernel
def mark_cell():
    for i, j in cell_type:
        if not is_solid(i, j):
            cell_type[i, j] = utils.AIR

    for i, j, ix, jx in particle_positions:
        if particle_type[i, j, ix, jx] == P_FLUID:
            pos = particle_positions[i, j, ix, jx]
            idx = ti.cast(ti.floor(pos / vec2(grid_x, grid_y)), ti.i32)

            if not is_solid(idx[0], idx[1]):
                cell_type[idx] = utils.FLUID


@ti.func
def backtrace(p, dt):
    # rk2 backtrace
    p_mid = p - 0.5 * dt * sample_velocity(p, u, v)
    p -= dt * sample_velocity(p_mid, u, v)

    return p


@ti.func
def semi_Largrange(x, x_temp, stagger, dt):
    m, n = x.shape
    for i, j in ti.ndrange(m, n):
        pos = (vec2(i, j) + stagger) * vec2(grid_x, grid_y)
        source_pos = backtrace(pos, dt)
        x_temp[i, j] = sample_bilinear(x, source_pos, stagger)


@ti.kernel
def advection_kernel(dt: ti.f32):
    semi_Largrange(u, u_temp, vec2(0, 0.5), dt)
    semi_Largrange(v, v_temp, vec2(0.5, 0), dt)


def advection(dt):
    advection_kernel(dt)
    u.copy_from(u_temp)
    v.copy_from(v_temp)


@ti.func
def Bspline(x):
    r = abs(x)
    res = 0.0

    if 0.0 <= r < 0.5:
        res = 0.75 - r**2
    elif 0.5 <= r < 1.5:
        res = 0.5 * (1.5 - r)**2

    return res


@ti.func
def Bspline_grad(x):
    r = abs(x)
    res = 0.0

    if 0.0 <= r < 0.5:
        res = -2 * x
    elif 0.5 <= x < 1.5:
        res = x - 1.5
    elif -1.5 < x <= -0.5:
        res = x + 1.5

    return res


@ti.func
def gather_vp(grid_v, grid_vlast, xp, stagger):
    inv_dx = vec2(1.0 / grid_x, 1.0 / grid_y).cast(ti.f32)
    base, _ = pos_to_stagger_idx(xp, stagger)

    v_pic = 0.0
    v_flip = 0.0

    for i in ti.static(range(-1, 3)):
        for j in ti.static(range(-1, 3)):
            I = vec2(i, j)
            pos = base + I + stagger
            fx = xp * inv_dx - pos
            weight = Bspline(fx[0]) * Bspline(fx[1])
            v_pic += weight * grid_v[base + I]
            v_flip += weight * (grid_v[base + I] - grid_vlast[base + I])

    return v_pic, v_flip


@ti.func
def gather_cp(grid_v, xp, stagger):
    inv_dx = vec2(1.0 / grid_x, 1.0 / grid_y).cast(ti.f32)
    base, _ = pos_to_stagger_idx(xp, stagger)

    cp = vec2(0.0, 0.0)

    for i in ti.static(range(-1, 3)):
        for j in ti.static(range(-1, 3)):
            I = vec2(i, j)
            pos = base + I + stagger
            fx = xp * inv_dx - pos
            gradweight = vec2(
                Bspline_grad(fx[0]) * Bspline(fx[1]),
                Bspline(fx[0]) * Bspline_grad(fx[1]))
            cp += gradweight * grid_v[base + I]

    return cp


@ti.kernel
def G2P():
    stagger_u = vec2(0.0, 0.5)
    stagger_v = vec2(0.5, 0.0)
    for p in ti.grouped(particle_positions):
        if particle_type[p] == P_FLUID:
            # update velocity
            xp = particle_positions[p]
            u_pic, u_flip = gather_vp(u, u_last, xp, stagger_u)
            v_pic, v_flip = gather_vp(v, v_last, xp, stagger_v)

            new_v_pic = vec2(u_pic, v_pic)

            if ti.static(algorithm == 'FLIP/PIC'):
                new_v_flip = particle_velocities[p] + vec2(u_flip, v_flip)

                particle_velocities[p] = FLIP_blending * new_v_flip + (
                    1 - FLIP_blending) * new_v_pic
            elif ti.static(algorithm == 'APIC'):
                particle_velocities[p] = new_v_pic
                cp_x[p] = gather_cp(u, xp, stagger_u)
                cp_y[p] = gather_cp(v, xp, stagger_v)


@ti.func
def scatter_vp(grid_v, grid_m, xp, vp, stagger):
    inv_dx = vec2(1.0 / grid_x, 1.0 / grid_y).cast(ti.f32)
    base, _ = pos_to_stagger_idx(xp, stagger)

    for i in ti.static(range(-1, 3)):
        for j in ti.static(range(-1, 3)):
            I = vec2(i, j)
            pos = base + I + stagger
            fx = xp * inv_dx - pos
            weight = Bspline(fx[0]) * Bspline(fx[1])
            grid_v[base + I] += weight * vp
            grid_m[base + I] += weight


@ti.func
def scatter_vp_apic(grid_v, grid_m, xp, vp, cp, stagger):
    inv_dx = vec2(1.0 / grid_x, 1.0 / grid_y).cast(ti.f32)
    base, _ = pos_to_stagger_idx(xp, stagger)

    for i in ti.static(range(-1, 3)):
        for j in ti.static(range(-1, 3)):
            I = vec2(i, j)
            pos = base + I + stagger
            fx = xp * inv_dx - pos
            weight = Bspline(fx[0]) * Bspline(fx[1])
            grid_v[base + I] += weight * (vp + cp.dot(fx))
            grid_m[base + I] += weight


@ti.kernel
def P2G():
    stagger_u = vec2(0.0, 0.5)
    stagger_v = vec2(0.5, 0.0)
    for p in ti.grouped(particle_positions):
        if particle_type[p] == P_FLUID:
            xp = particle_positions[p]

            if ti.static(algorithm == 'FLIP/PIC'):
                scatter_vp(u, u_weight, xp, particle_velocities[p][0],
                           stagger_u)
                scatter_vp(v, v_weight, xp, particle_velocities[p][1],
                           stagger_v)
            elif ti.static(algorithm == 'APIC'):
                scatter_vp_apic(u, u_weight, xp, particle_velocities[p][0],
                                cp_x[p], stagger_u)
                scatter_vp_apic(v, v_weight, xp, particle_velocities[p][1],
                                cp_y[p], stagger_v)


@ti.kernel
def grid_norm():
    for i, j in u:
        if u_weight[i, j] > 0:
            u[i, j] = u[i, j] / u_weight[i, j]

    for i, j in v:
        if v_weight[i, j] > 0:
            v[i, j] = v[i, j] / v_weight[i, j]


def onestep(dt):
    apply_gravity(dt)
    enforce_boundary()

    extrapolate_velocity()
    enforce_boundary()

    solve_pressure(dt)
    apply_pressure(dt)
    enforce_boundary()

    extrapolate_velocity()
    enforce_boundary()

    if algorithm == 'FLIP/PIC' or algorithm == 'APIC':
        G2P()
        advect_particles(dt)
        mark_cell()

        u.fill(0.0)
        v.fill(0.0)
        u_weight.fill(0.0)
        v_weight.fill(0.0)

        P2G()
        grid_norm()
        enforce_boundary()

        u_last.copy_from(u)
        v_last.copy_from(v)

    else:
        update_particle_velocities(dt)
        advect_particles(dt)
        mark_cell()

        advection(dt)
        enforce_boundary()


def simulation(max_time, max_step):
    dt = 0.01
    t = 0
    step = 1

    while step < max_step and t < max_time:
        render()

        for i in range(substeps):
            onestep(dt)

            pv = particle_velocities.to_numpy()
            max_vel = np.max(np.linalg.norm(pv, 2, axis=1))

            print("step = {}, substeps = {}, time = {}, dt = {}, maxv = {}".
                  format(step, i, t, dt, max_vel))

            t += dt
            # update dt with CFL
            # dt = 5 * grid_x / max_vel

        step += 1


def main():
    init()
    t0 = time.time()
    simulation(40, 240)
    t1 = time.time()
    print("simulation elapsed time = {} seconds".format(t1 - t0))

    video_manager.make_video(gif=False, mp4=True)


if __name__ == "__main__":
    main()
