"""
This project will include the normal optimization algorithms for the DeepLearning :
1:gradient descend
2:Ada grad
3:RMS prop
4:Momentum
5:Nesterov
6:Adam
All of the codes are written by a material cunt .
"""
import numpy as np
import matplotlib.pyplot as plt


class OptimizationAlgorithm:
    def __init__(self, obj_func, num_var, lower_bound, upper_bound, epoch, analytical_grad=None):
        """
        This is the initialization of this problem , and it is usually defined as a minimizing problem .
        Parameters
        ----------
        obj_func: Objective function
        num_var: number of variables for this function
        lower_bound:  the lower bound for the variables . Obviously , this is a list .
        upper_bound:  the upper bound for the variables . Obviously , this is a list .
        epoch: the number of the iteration.
        analytical_grad: the analytical grad for the function if given .
        """

        self.obj_func = obj_func
        self.num_var = num_var
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.epoch = epoch
        if analytical_grad:
            self.analytical_grad = analytical_grad
            self.has_analytical_grad = True
        else:
            self.has_analytical_grad = False

    @staticmethod
    def numerical_gradient(f, x, h=1e-8):
        """

        Parameters
        ----------
        f:objective function
        x:at where the grad is obtained
        h:the step

        Returns
        -------
        return the corresponding gradient
        """
        grad = np.zeros_like(x)

        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            tmp_val = x[idx]
            x[idx] = float(tmp_val) + h
            fxh1 = f(x)  # f(x+h)
            x[idx] = tmp_val - h
            fxh2 = f(x)  # f(x-h)
            grad[idx] = (fxh1 - fxh2) / (2 * h)
            x[idx] = tmp_val  # 还原值
            it.iternext()

        return grad

    def gradient_descend(self, grad_method, lr=1e-3, display_process=False):
        """
        Use the general gradient descend method to obtain the optimal solution.
        Parameters
        ----------
        grad_method: The method to obtain gradient , by analytical or numerical
        lr: The learning rate .
        display_process: whether display the iteration process.

        Returns
        -------
        Return the x , where the objective function gets the minimum.
        """
        x = np.random.uniform(low=self.lower_bound, high=self.upper_bound)
        for i in range(self.epoch):
            if grad_method == "analytical" and self.has_analytical_grad:
                # use the analytical method to obtain the gradient
                grad = self.analytical_grad(x)
            if grad_method == "numerical":
                # use the numerical method to obtain the gradient
                grad = self.numerical_gradient(self.obj_func, x)
            x -= lr*grad
            if display_process:
                if i % 50 == 0:
                    print(f"epoch: {i}/{self.epoch},  x={x},  f={self.obj_func(x)}")

        return x

    def ada_grad(self, grad_method, lr=1e-1, display_process=False):
        """
        Use the adaptive gradient descend method to obtain the optimal solution.
        For the sumation of the grad will accumulate , so the step for each iteration
        will approach the 0 . In the end , the iteration will terminate .
        Parameters
        ----------
        grad_method: The method to obtain gradient , by analytical or numerical
        lr: The learning rate .
        display_process: whether display the iteration process.

        Returns
        -------
        Return the x , where the objective function gets the minimum.
        """
        # for the ada grad algorithm , a list to record the history should be maintained
        gt = 0
        x = np.random.uniform(low=self.lower_bound, high=self.upper_bound)
        for i in range(self.epoch):
            if grad_method == "analytical" and self.has_analytical_grad:
                # use the analytical method to obtain the gradient
                grad = self.analytical_grad(x)
            if grad_method == "numerical":
                # use the numerical method to obtain the gradient
                grad = self.numerical_gradient(self.obj_func, x)
            # get the accumulative grad
            gt += np.sum(np.square(grad))
            # The Ada grad algorithm is implemented!
            x -= lr*grad/np.sqrt(gt+1e-9)
            if display_process:
                if i % 50 == 0:
                    print(f"epoch: {i}/{self.epoch},  x={x},  f={self.obj_func(x)}")

        return x

    def rms_prop(self, grad_method, lr=1e-2, display_process=False):
        """
        Use the rms prop method to obtain the optimal solution.
        For the sumation of the grad will accumulate , so the step for each iteration
        will approach the 0 . In the end , the iteration will terminate .
        Parameters
        ----------
        grad_method: The method to obtain gradient , by analytical or numerical
        lr: The learning rate .
        display_process: whether display the iteration process.

        Returns
        -------
        Return the x , where the objective function gets the minimum.
        """
        # for the rms prop algorithm , a sumation of the grad should be maintained .
        gt = 0
        beta = 0.9

        x = np.random.uniform(low=self.lower_bound, high=self.upper_bound)
        # self.w = np.array((x[0], x[1]))
        for i in range(self.epoch):
            if grad_method == "analytical" and self.has_analytical_grad:
                # use the analytical method to obtain the gradient
                grad = self.analytical_grad(x)
            if grad_method == "numerical":
                # use the numerical method to obtain the gradient
                grad = self.numerical_gradient(self.obj_func, x)
            # self.w = np.vstack((self.w, (x[0],x[1])))
            # get the accumulative grad
            gt = beta*gt+(1-beta)*np.sum(np.square(grad))
            # The Ada grad algorithm is implemented!
            x -= lr * grad / (np.sqrt(gt) + 1e-9)
            if display_process:
                if i % 50 == 0:
                    print(f"epoch: {i}/{self.epoch},  x={x},  f={self.obj_func(x)}")

        return x

    def momentum(self, grad_method, lr=1e-3, display_process=False):
        """
        Use the momentum modified general gradient descend method to obtain the optimal solution.
        The momentum method will use the momentum to avoid the local optimal
        Parameters
        ----------
        grad_method: The method to obtain gradient , by analytical or numerical
        lr: The learning rate .
        display_process: whether display the iteration process.

        Returns
        -------
        Return the x , where the objective function gets the minimum.
        """
        x = np.random.uniform(low=self.lower_bound, high=self.upper_bound)
        # the initial velocity
        velocity = np.zeros_like(x)
        momentum_factor = 0.1
        for i in range(self.epoch):
            if grad_method == "analytical" and self.has_analytical_grad:
                # use the analytical method to obtain the gradient
                grad = self.analytical_grad(x)
            if grad_method == "numerical":
                # use the numerical method to obtain the gradient
                grad = self.numerical_gradient(self.obj_func, x)
            # update the velocity
            velocity = velocity*momentum_factor-lr*grad
            x = x+velocity
            if display_process:
                if i % 50 == 0:
                    print(f"epoch: {i}/{self.epoch},  x={x},  f={self.obj_func(x)}")

        return x

    def nesterov(self, grad_method, lr=1e-3, display_process=False):
        """
        Use the momentum modified general gradient descend method to obtain the optimal solution.
        The momentum method will use the momentum to avoid the local optimal.
        Nesterov method is a kind of acceleration method that step ahead a bit before
        the gradient descend .
        Parameters
        ----------
        grad_method: The method to obtain gradient , by analytical or numerical
        lr: The learning rate .
        display_process: whether display the iteration process.

        Returns
        -------
        Return the x , where the objective function gets the minimum.
        """
        x = np.random.uniform(low=self.lower_bound, high=self.upper_bound)
        # the initial velocity
        velocity = np.zeros_like(x)
        momentum_factor = 0.1

        # then all of the grads should be acquired in the accelerated_x rather than x
        for i in range(self.epoch):
            # accelerate the x first before obtaining the grad
            accelerated_x = x + velocity * momentum_factor
            if grad_method == "analytical" and self.has_analytical_grad:
                # use the analytical method to obtain the gradient
                grad = self.analytical_grad(accelerated_x)
            if grad_method == "numerical":
                # use the numerical method to obtain the gradient
                grad = self.numerical_gradient(self.obj_func, accelerated_x)
            # update the velocity , this is the same as the momentum method
            velocity = velocity * momentum_factor - lr * grad
            x = x + velocity
            if display_process:
                if i % 50 == 0:
                    print(f"epoch: {i}/{self.epoch},  x={x},  f={self.obj_func(x)}")

        return x

    def adam(self, grad_method, lr=1e-1, display_process=False):
        """
        This is the implementation for the adam algorithm , which
        is the combination of the ada grad and momentum algorithm.
        Parameters
        ----------
        grad_method: The method to obtain gradient , by analytical or numerical
        lr: The learning rate .
        display_process: whether display the iteration process.

        Returns
        -------
        Return the x , where the objective function gets the minimum.

        """
        # the declining rate for the moment estimation
        rho1, rho2 = 0.9, 0.999
        delta = 1e-8
        gt = 0
        x = np.random.uniform(low=self.lower_bound, high=self.upper_bound)
        # initialize the first moment and the second moment
        s = np.zeros_like(x)
        r = np.zeros_like(x)
        for i in range(self.epoch):
            if grad_method == "analytical" and self.has_analytical_grad:
                # use the analytical method to obtain the gradient
                grad = self.analytical_grad(x)
            if grad_method == "numerical":
                # use the numerical method to obtain the gradient
                grad = self.numerical_gradient(self.obj_func, x)
            # update the first and the second moment
            s = rho1*s+(1-rho1)*grad
            r = rho2*r+(1-rho2)*np.square(grad)
            # get the partial first and second moment
            s_hat = s/(1-rho1**(i+1))
            r_hat = r/(1-rho1**(i+1))
            # use the partial first and second moment to modify the GD method
            x -= lr * s_hat / (np.sqrt(r_hat)+delta)
            if display_process:
                if i % 50 == 0:
                    print(f"epoch: {i}/{self.epoch},  x={x},  f={self.obj_func(x)}")

        return x


def draw_process(w):
        x = np.arange(-100, 100, 0.1)
        x = np.arange(-100, 100, 0.1)
        [x, y] = np.meshgrid(x, x)
        f = x**2/20+y**2
        plt.contour(x, y, f, 20)
        plt.plot(w[:, 0], w[:, 1], 'g*', w[:, 0], w[:, 1])
        plt.show()

if __name__ == '__main__':
    obj_func = lambda x: x[0]**2/20+x[1]**2
    # solver = OptimizationAlgorithm(obj_func=obj_func, num_var=2, lower_bound=[-100, -100], upper_bound=[100, 100], epoch=1000000)
    # optimal_x = solver.gradient_descend(grad_method="numerical", display_process=True)
    solver = OptimizationAlgorithm(obj_func=obj_func, num_var=2,
                                   lower_bound=[-1000, -1000], upper_bound=[1000, 1000],
                                   epoch=200000, analytical_grad=lambda x: np.array([x[0]/10, 2*x[1]]))
    optimal_x = solver.adam(grad_method="numerical", display_process=True)
    print(f"optimal_x:{optimal_x}")
    print(f"minimum function value {obj_func(optimal_x)}")
    # draw_process(solver.w)