import taichi as ti
import utils


@ti.data_oriented
class MICPCGSolver:
    def __init__(self, m, n, u, v, cell_type, MIC_blending=0.0):
        self.m = m
        self.n = n
        self.u = u
        self.v = v
        self.cell_type = cell_type
        self.MIC_blending = MIC_blending

        # rhs of linear system
        self.b = ti.field(dtype=ti.f32, shape=(self.m, self.n))

        # lhs of linear system
        self.Adiag = ti.field(dtype=ti.f32, shape=(self.m, self.n))
        self.Ax = ti.field(dtype=ti.f32, shape=(self.m, self.n))
        self.Ay = ti.field(dtype=ti.f32, shape=(self.m, self.n))

        # cg var
        self.p = ti.field(dtype=ti.f32, shape=(self.m, self.n))
        self.r = ti.field(dtype=ti.f32, shape=(self.m, self.n))
        self.s = ti.field(dtype=ti.f32, shape=(self.m, self.n))
        self.As = ti.field(dtype=ti.f32, shape=(self.m, self.n))
        self.sum = ti.field(dtype=ti.f32, shape=())
        self.alpha = ti.field(dtype=ti.f32, shape=())
        self.beta = ti.field(dtype=ti.f32, shape=())

        # MIC precondition
        self.precon = ti.field(dtype=ti.f32, shape=(self.m, self.n))
        self.z = ti.field(dtype=ti.f32, shape=(self.m, self.n))
        self.q = ti.field(dtype=ti.f32, shape=(self.m, self.n))

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

    @ti.kernel
    def preconditioner_init(self):
        sigma = 0.25  # safety constant

        for _ in range(1):  # force serial
            for i, j in ti.ndrange(self.m, self.n):
                if self.cell_type[i, j] == utils.FLUID:
                    e = self.Adiag[i, j] - (
                        self.Ax[i - 1, j] * self.precon[i - 1, j])**2 - (
                            self.Ay[i, j - 1] *
                            self.precon[i, j - 1])**2 - self.MIC_blending * (
                                self.Ax[i - 1, j] * self.Ay[i - 1, j] *
                                self.precon[i - 1, j]**2 + self.Ay[i, j - 1] *
                                self.Ax[i, j - 1] * self.precon[i, j - 1]**2)

                    if e < sigma * self.Adiag[i, j]:
                        e = self.Adiag[i, j]

                    self.precon[i, j] = 1 / ti.sqrt(e)

    def system_init(self, scale_A, scale_b):
        self.b.fill(0)
        self.Adiag.fill(0.0)
        self.Ax.fill(0.0)
        self.Ay.fill(0.0)
        self.precon.fill(0.0)

        self.system_init_kernel(scale_A, scale_b)
        self.preconditioner_init()

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
            # z0 = M^-1r0
            self.q.fill(0.0)
            self.z.fill(0.0)
            self.applyPreconditioner()

            # s0 = z0
            self.s.copy_from(self.z)

            # zTr
            self.reduce(self.z, self.r)
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
                self.reduce(self.r, self.r)
                rTr = self.sum[None]
                if rTr < init_rTr * tol:
                    break

                # z = M^-1r
                self.q.fill(0.0)
                self.z.fill(0.0)
                self.applyPreconditioner()

                self.reduce(self.z, self.r)
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
                self.s[i, j] = self.z[i, j] + self.beta[None] * self.s[i, j]

    @ti.kernel
    def applyPreconditioner(self):
        # # first solve Lq = r
        for _ in range(1):
            for i, j in ti.ndrange(self.m, self.n):
                if self.cell_type[i, j] == utils.FLUID:
                    t = self.r[i, j] - self.Ax[i - 1, j] * self.precon[
                        i - 1, j] * self.q[i - 1, j] - self.Ay[
                            i, j - 1] * self.precon[i, j - 1] * self.q[i,
                                                                       j - 1]

                    self.q[i, j] = t * self.precon[i, j]

        # next solve LTz = q
        for _ in range(1):
            for ix, iy in ti.ndrange(self.m, self.n):
                i = self.m - 1 - ix
                j = self.n - 1 - iy

                if self.cell_type[i, j] == utils.FLUID:
                    t = self.q[i, j] - self.Ax[i, j] * self.precon[
                        i, j] * self.z[i + 1, j] - self.Ay[i, j] * self.precon[
                            i, j] * self.z[i, j + 1]

                    self.z[i, j] = t * self.precon[i, j]