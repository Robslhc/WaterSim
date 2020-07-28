import taichi as ti
from CGSolver import CGSolver
import numpy as np
from utils import ColorMap, vec2, vec3, clamp
import utils
import random

ti.init(arch=ti.cpu, default_fp=ti.f32)

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

# params in render
screen_res = (400, 400 * n // m)
bwrR = ColorMap(1.0, .25, 1, .5)
bwrG = ColorMap(1.0, .5, .5, .5)
bwrB = ColorMap(1.0, 1, .25, .5)
color_buffer = ti.Vector(3, dt=ti.f32, shape=screen_res)
gui = ti.GUI("watersim2D", screen_res)

# cell type
cell_type = ti.var(dt=ti.i32, shape=(m, n))

# velocity field
u = ti.var(dt=ti.f32, shape=(m + 1, n))
v = ti.var(dt=ti.f32, shape=(m, n + 1))
u_temp = ti.var(dt=ti.f32, shape=(m + 1, n))
v_temp = ti.var(dt=ti.f32, shape=(m, n + 1))

# pressure field
p = ti.var(dt=ti.f32, shape=(m, n))

#pressure solver
solver = CGSolver(m, n, u, v, cell_type)

# particle x and v
particle_positions = ti.Vector(2, dt=ti.f32, shape=(m, n, npar, npar))
particle_velocities = ti.Vector(2, dt=ti.f32, shape=(m, n, npar, npar))

# particle type
particle_type = ti.var(dt=ti.f32, shape=(m, n, npar, npar))
P_FLUID = 1
P_OTHER = 0

# extrap utils
valid = ti.var(dt=ti.i32, shape=(m + 1, n + 1))
valid_temp = ti.var(dt=ti.i32, shape=(m + 1, n + 1))

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

        for i, j in v:
            v[i, j] = 0.0

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
def update_particle(dt: ti.f32):
    for i, j, ix, jx in particle_velocities:
        if particle_type[i, j, ix, jx] == P_FLUID:
            pos = particle_positions[i, j, ix, jx]
            pv = sample_velocity(pos, u, v)
            pos += pv * dt

            if pos[0] < grid_x:  # left boundary
                pos[0] = grid_x
                pv[0] = 0
            if pos[0] > w - grid_x:  # right boundary
                pos[0] = w - grid_x
                pv[0] = 0
            if pos[1] < grid_y:  # bottom boundary
                pos[1] = grid_y
                pv[1] = 0
            if pos[1] > h - grid_y:  # top boundary
                pos[1] = h - grid_y
                pv[1] = 0

            particle_velocities[i, j, ix, jx] = pv
            particle_positions[i, j, ix, jx] = pos


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

    update_particle(dt)
    mark_cell()

    advection(dt)
    enforce_boundary()


def simulation(max_time, max_step):
    dt = 0.025
    t = 0
    step = 1

    render()

    while step < max_step and t < max_time:
        onestep(dt)

        render()

        pv = particle_velocities.to_numpy()
        max_vel = np.max(np.linalg.norm(pv, 2, axis=1))

        print("step = {}, time = {}, dt = {}, maxv = {}".format(
            step, t, dt, max_vel))

        t += dt
        step += 1

        # update dt with CFL
        # dt = 0.8 * grid_x / max_vel


def main():
    init()
    simulation(20, 480)
    video_manager.make_video(gif=False, mp4=True)


if __name__ == "__main__":
    main()
