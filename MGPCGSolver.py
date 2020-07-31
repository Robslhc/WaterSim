import taichi as ti
import utils


@ti.data_oriented
class MGPCGSolver:
    def __init__(self,
                 m,
                 n,
                 u,
                 v,
                 cell_type,
                 multigrid_level=4,
                 pre_and_post_smoothing=2,
                 bottom_smoothing=10):
        self.m = m
        self.n = n
        self.u = u
        self.v = v
        self.cell_type = cell_type
        self.multigrid_level = multigrid_level
        self.pre_and_post_smoothing = pre_and_post_smoothing
        self.bottom_smoothing = bottom_smoothing

        # rhs of linear system
        self.b = ti.field(dtype=ti.f32, shape=(self.m, self.n))

        def grid_shape(l):
            return (self.m // 2**l, self.n // 2**l)

        # lhs of linear system and its corresponding form in coarse grids
        self.Adiag = [
            ti.field(dtype=ti.f32, shape=grid_shape(l))
            for l in range(self.multigrid_level)
        ]
        self.Ax = [
            ti.field(dtype=ti.f32, shape=grid_shape(l))
            for l in range(self.multigrid_level)
        ]
        self.Ay = [
            ti.field(dtype=ti.f32, shape=grid_shape(l))
            for l in range(self.multigrid_level)
        ]

        # grid type
        self.grid_type = [
            ti.field(dtype=ti.i32, shape=grid_shape(l))
            for l in range(self.multigrid_level)
        ]

        # pcg var
        self.r = [
            ti.field(dtype=ti.f32, shape=grid_shape(l))
            for l in range(self.multigrid_level)
        ]
        self.z = [
            ti.field(dtype=ti.f32, shape=grid_shape(l))
            for l in range(self.multigrid_level)
        ]

        self.p = ti.field(dtype=ti.f32, shape=(self.m, self.n))
        self.s = ti.field(dtype=ti.f32, shape=(self.m, self.n))
        self.As = ti.field(dtype=ti.f32, shape=(self.m, self.n))
        self.sum = ti.field(dtype=ti.f32, shape=())
        self.alpha = ti.field(dtype=ti.f32, shape=())
        self.beta = ti.field(dtype=ti.f32, shape=())

    @ti.kernel
    def system_init_kernel(self, scale_A: ti.f32, scale_b: ti.f32):
        #define right hand side of linear system
        for i, j in ti.ndrange(self.m, self.n):
            if self.cell_type[i, j] == utils.FLUID:
                self.b[i,
                       j] = -1 * scale_b * (self.u[i + 1, j] - self.u[i, j] +
                                            self.v[i, j + 1] - self.v[i, j])

        #modify right hand side of linear system to account for solid velocities
        #currently hard code solid velocities to zero
        for i, j in ti.ndrange(self.m, self.n):
            if self.cell_type[i, j] == utils.FLUID:
                if self.cell_type[i - 1, j] == utils.SOLID:
                    self.b[i, j] -= scale_b * (self.u[i, j] - 0)
                if self.cell_type[i + 1, j] == utils.SOLID:
                    self.b[i, j] += scale_b * (self.u[i + 1, j] - 0)

                if self.cell_type[i, j - 1] == utils.SOLID:
                    self.b[i, j] -= scale_b * (self.v[i, j] - 0)
                if self.cell_type[i, j + 1] == utils.SOLID:
                    self.b[i, j] += scale_b * (self.v[i, j + 1] - 0)

        # define left handside of linear system
        for i, j in ti.ndrange(self.m, self.n):
            if self.cell_type[i, j] == utils.FLUID:
                if self.cell_type[i - 1, j] == utils.FLUID:
                    self.Adiag[0][i, j] += scale_A
                if self.cell_type[i + 1, j] == utils.FLUID:
                    self.Adiag[0][i, j] += scale_A
                    self.Ax[0][i, j] = -scale_A
                elif self.cell_type[i + 1, j] == utils.AIR:
                    self.Adiag[0][i, j] += scale_A

                if self.cell_type[i, j - 1] == utils.FLUID:
                    self.Adiag[0][i, j] += scale_A
                if self.cell_type[i, j + 1] == utils.FLUID:
                    self.Adiag[0][i, j] += scale_A
                    self.Ay[0][i, j] = -scale_A
                elif self.cell_type[i, j + 1] == utils.AIR:
                    self.Adiag[0][i, j] += scale_A

    @ti.kernel
    def gridtype_init(self, l: ti.template()):
        for i, j in self.grid_type[l]:
            # if i == 0 or i == self.m // (2**l) - 1 or j == 0 or j == self.n // (2 ** l) - 1:
            #     self.grid_type[l][i, j] = utils.SOLID

            i2 = i * 2
            j2 = j * 2

            if self.grid_type[l - 1][i2, j2] == utils.AIR or self.grid_type[
                    l - 1][i2, j2 + 1] == utils.AIR or self.grid_type[l - 1][
                        i2 + 1,
                        j2] == utils.AIR or self.grid_type[l -
                                                           1][i2 + 1, j2 +
                                                              1] == utils.AIR:
                self.grid_type[l][i, j] = utils.AIR
            else:
                if self.grid_type[l - 1][
                        i2, j2] == utils.FLUID or self.grid_type[l - 1][
                            i2,
                            j2 + 1] == utils.FLUID or self.grid_type[l - 1][
                                i2 + 1, j2] == utils.FLUID or self.grid_type[
                                    l - 1][i2 + 1, j2 + 1] == utils.FLUID:
                    self.grid_type[l][i, j] = utils.FLUID
                else:
                    self.grid_type[l][i, j] = utils.SOLID

    @ti.kernel
    def preconditioner_init(self, scale: ti.f32, l: ti.template()):
        scale = scale / (2**l * 2**l)

        for i, j in self.grid_type[l]:
            if self.grid_type[l][i, j] == utils.FLUID:
                if self.grid_type[l][i - 1, j] == utils.FLUID:
                    self.Adiag[l][i, j] += scale
                if self.grid_type[l][i + 1, j] == utils.FLUID:
                    self.Adiag[l][i, j] += scale
                    self.Ax[l][i, j] = -scale
                elif self.grid_type[l][i + 1, j] == utils.AIR:
                    self.Adiag[l][i, j] += scale

                if self.grid_type[l][i, j - 1] == utils.FLUID:
                    self.Adiag[l][i, j] += scale
                if self.grid_type[l][i, j + 1] == utils.FLUID:
                    self.Adiag[l][i, j] += scale
                    self.Ay[l][i, j] = -scale
                elif self.grid_type[l][i, j + 1] == utils.AIR:
                    self.Adiag[l][i, j] += scale

    def system_init(self, scale_A, scale_b):
        self.b.fill(0.0)

        for l in range(self.multigrid_level):
            self.Adiag[l].fill(0.0)
            self.Ax[l].fill(0.0)
            self.Ay[l].fill(0.0)

        self.system_init_kernel(scale_A, scale_b)
        self.grid_type[0].copy_from(self.cell_type)

        for l in range(1, self.multigrid_level):
            self.gridtype_init(l)
            self.preconditioner_init(scale_A, l)

    @ti.func
    def neighbor_sum(self, Ax, Ay, z, nx, ny, i, j):
        Az = Ax[(i - 1 + nx) % nx, j] * z[(i - 1 + nx) % nx, j] + Ax[i, j] * z[
            (i + 1) % nx, j] + Ay[i, (j - 1 + ny) % ny] * z[
                i, (j - 1 + ny) % ny] + Ay[i, j] * z[i, (j + 1) % ny]

        return Az

    @ti.kernel
    def smooth(self, l: ti.template(), phase: ti.i32):
        # phase: red/black Gauss-Seidel phase
        for i, j in self.r[l]:
            if self.grid_type[l][i, j] == utils.FLUID and (i + j) & 1 == phase:
                self.z[l][i, j] = (self.r[l][i, j] - self.neighbor_sum(
                    self.Ax[l], self.Ay[l], self.z[l], self.m //
                    (2**l), self.n // (2**l), i, j)) / self.Adiag[l][i, j]

    @ti.kernel
    def restrict(self, l: ti.template()):
        for i, j in self.r[l]:
            if self.grid_type[l][i, j] == utils.FLUID:
                Az = self.Adiag[l][i, j] * self.z[l][i, j]
                Az += self.neighbor_sum(self.Ax[l], self.Ay[l], self.z[l],
                                        self.m // (2**l), self.n // (2**l), i,
                                        j)
                res = self.r[l][i, j] - Az

                self.r[l + 1][i // 2, j // 2] += 0.25 * res

    @ti.kernel
    def prolongate(self, l: ti.template()):
        for i, j in self.z[l]:
            self.z[l][i, j] += self.z[l + 1][i // 2, j // 2]

    def v_cycle(self):
        self.z[0].fill(0.0)
        for l in range(self.multigrid_level - 1):
            for i in range(self.pre_and_post_smoothing):
                self.smooth(l, 0)
                self.smooth(l, 1)

            self.r[l + 1].fill(0.0)
            self.z[l + 1].fill(0.0)
            self.restrict(l)

        # solve Az = r on the coarse grid
        for i in range(self.bottom_smoothing):
            self.smooth(self.multigrid_level - 1, 0)
            self.smooth(self.multigrid_level - 1, 1)

        for l in reversed(range(self.multigrid_level - 1)):
            self.prolongate(l)
            for i in range(self.pre_and_post_smoothing):
                self.smooth(l, 1)
                self.smooth(l, 0)

    def solve(self, max_iters):
        tol = 1e-12

        self.p.fill(0.0)
        self.As.fill(0.0)
        self.s.fill(0.0)
        self.r[0].copy_from(self.b)

        self.reduce(self.r[0], self.r[0])
        init_rTr = self.sum[None]

        print("init rTr = {}".format(init_rTr))

        if init_rTr < tol:
            print("Converged: init rtr = {}".format(init_rTr))
        else:
            # p0 = 0
            # r0 = b - Ap0 = b
            # z0 = M^-1r0
            # self.z.fill(0.0)
            self.v_cycle()

            # s0 = z0
            self.s.copy_from(self.z[0])

            # zTr
            self.reduce(self.z[0], self.r[0])
            old_zTr = self.sum[None]

            iteration = 0

            for i in range(max_iters):
                # alpha = zTr / sAs
                self.compute_As()
                self.reduce(self.s, self.As)
                sAs = self.sum[None]
                self.alpha[None] = old_zTr / sAs

                # p = p + alpha * s
                self.update_p()

                # r = r - alpha * As
                self.update_r()

                # check for convergence
                self.reduce(self.r[0], self.r[0])
                rTr = self.sum[None]
                if rTr < init_rTr * tol:
                    break

                # z = M^-1r
                self.v_cycle()

                self.reduce(self.z[0], self.r[0])
                new_zTr = self.sum[None]

                # beta = zTrnew / zTrold
                self.beta[None] = new_zTr / old_zTr

                # s = z + beta * s
                self.update_s()
                old_zTr = new_zTr
                iteration = i

                # if iteration % 100 == 0:
                #     print("iter {}, res = {}".format(iteration, rTr))

            print("Converged to {} in {} iterations".format(rTr, iteration))

    @ti.kernel
    def reduce(self, p: ti.template(), q: ti.template()):
        self.sum[None] = 0.0
        for i, j in ti.ndrange(self.m, self.n):
            if self.cell_type[i, j] == utils.FLUID:
                self.sum[None] += p[i, j] * q[i, j]

    @ti.kernel
    def compute_As(self):
        for i, j in ti.ndrange(self.m, self.n):
            if self.cell_type[i, j] == utils.FLUID:
                self.As[i, j] = self.Adiag[0][i, j] * self.s[
                    i, j] + self.Ax[0][i - 1, j] * self.s[
                        i - 1, j] + self.Ax[0][i, j] * self.s[
                            i + 1, j] + self.Ay[0][i, j - 1] * self.s[
                                i, j - 1] + self.Ay[0][i, j] * self.s[i, j + 1]

    @ti.kernel
    def update_p(self):
        for i, j in ti.ndrange(self.m, self.n):
            if self.cell_type[i, j] == utils.FLUID:
                self.p[i, j] = self.p[i, j] + self.alpha[None] * self.s[i, j]

    @ti.kernel
    def update_r(self):
        for i, j in ti.ndrange(self.m, self.n):
            if self.cell_type[i, j] == utils.FLUID:
                self.r[0][i,
                          j] = self.r[0][i,
                                         j] - self.alpha[None] * self.As[i, j]

    @ti.kernel
    def update_s(self):
        for i, j in ti.ndrange(self.m, self.n):
            if self.cell_type[i, j] == utils.FLUID:
                self.s[i, j] = self.z[0][i, j] + self.beta[None] * self.s[i, j]
