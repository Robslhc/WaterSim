import taichi as ti
import utils


@ti.data_oriented
class CGSolver:
    def __init__(self, m, n, u, v, cell_type):
        self.m = m
        self.n = n
        self.u = u
        self.v = v
        self.cell_type = cell_type

        # rhs of linear system
        self.b = ti.var(dt=ti.f32, shape=(self.m, self.n))

        # lhs of linear system
        self.Adiag = ti.var(dt=ti.f32, shape=(self.m, self.n))
        self.Ax = ti.var(dt=ti.f32, shape=(self.m, self.n))
        self.Ay = ti.var(dt=ti.f32, shape=(self.m, self.n))

        # cg var
        self.p = ti.var(dt=ti.f32, shape=(self.m, self.n))
        self.r = ti.var(dt=ti.f32, shape=(self.m, self.n))
        self.s = ti.var(dt=ti.f32, shape=(self.m, self.n))
        self.As = ti.var(dt=ti.f32, shape=(self.m, self.n))
        self.sum = ti.var(dt=ti.f32, shape=())
        self.alpha = ti.var(dt=ti.f32, shape=())
        self.beta = ti.var(dt=ti.f32, shape=())

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
                    self.Adiag[i, j] += scale_A
                if self.cell_type[i + 1, j] == utils.FLUID:
                    self.Adiag[i, j] += scale_A
                    self.Ax[i, j] = -scale_A
                elif self.cell_type[i + 1, j] == utils.AIR:
                    self.Adiag[i, j] += scale_A

                if self.cell_type[i, j - 1] == utils.FLUID:
                    self.Adiag[i, j] += scale_A
                if self.cell_type[i, j + 1] == utils.FLUID:
                    self.Adiag[i, j] += scale_A
                    self.Ay[i, j] = -scale_A
                elif self.cell_type[i, j + 1] == utils.AIR:
                    self.Adiag[i, j] += scale_A

    def system_init(self, scale_A, scale_b):
        self.b.fill(0)
        self.Adiag.fill(0.0)
        self.Ax.fill(0.0)
        self.Ay.fill(0.0)

        self.system_init_kernel(scale_A, scale_b)

    def solve(self, max_iters):
        tol = 1e-12

        self.p.fill(0.0)
        self.As.fill(0.0)
        self.s.fill(0.0)
        self.r.copy_from(self.b)

        self.reduce(self.r, self.r)
        init_rTr = self.sum[None]

        print("init rTr = {}".format(init_rTr))

        if init_rTr < tol:
            print("Converged: init rtr = {}".format(init_rTr))
        else:
            # p0 = 0
            # r0 = b - Ap0 = b
            # s0 = r0
            self.s.copy_from(self.r)
            old_rTr = init_rTr
            iteration = 0

            for i in range(max_iters):
                # alpha = rTr / sAs
                self.compute_As()
                self.reduce(self.s, self.As)
                sAs = self.sum[None]
                self.alpha[None] = old_rTr / sAs

                # p = p + alpha * s
                self.update_p()

                # r = r - alpha * As
                self.update_r()

                # check for convergence
                self.reduce(self.r, self.r)
                rTr = self.sum[None]
                if rTr < init_rTr * tol:
                    break

                new_rTr = rTr
                self.beta[None] = new_rTr / old_rTr

                # s = r + beta * s
                self.update_s()
                old_rTr = new_rTr
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
                self.As[i, j] = self.Adiag[i, j] * self.s[i, j] + self.Ax[
                    i - 1, j] * self.s[i - 1, j] + self.Ax[i, j] * self.s[
                        i + 1, j] + self.Ay[i, j - 1] * self.s[
                            i, j - 1] + self.Ay[i, j] * self.s[i, j + 1]

    @ti.kernel
    def update_p(self):
        for i, j in ti.ndrange(self.m, self.n):
            if self.cell_type[i, j] == utils.FLUID:
                self.p[i, j] = self.p[i, j] + self.alpha[None] * self.s[i, j]

    @ti.kernel
    def update_r(self):
        for i, j in ti.ndrange(self.m, self.n):
            if self.cell_type[i, j] == utils.FLUID:
                self.r[i, j] = self.r[i, j] - self.alpha[None] * self.As[i, j]

    @ti.kernel
    def update_s(self):
        for i, j in ti.ndrange(self.m, self.n):
            if self.cell_type[i, j] == utils.FLUID:
                self.s[i, j] = self.r[i, j] + self.beta[None] * self.s[i, j]
