"""
This project will include the normal optimization algorithms for the DeepLearning :
1:gradient descend
2:Ada grad
3:RMS prop
4:Momentum
5:Neserov
6:Adam
All of the codes are written by a material cunt .  
"""
import numpy as np


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


if __name__ == '__main__':
    obj_func = lambda x: x[0]**2/20+x[1]**2
    #solver = OptimizationAlgorithm(obj_func=obj_func, num_var=2, lower_bound=[-100, -100], upper_bound=[100, 100], epoch=1000000)
    #optimal_x = solver.gradient_descend(grad_method="numerical", display_process=True)
    solver = OptimizationAlgorithm(obj_func=obj_func, num_var=2,
                                   lower_bound=[-100, -100], upper_bound=[100, 100],
                                   epoch=1000000, analytical_grad=lambda x: np.array([x[0]/10, 2*x[1]]))
    optimal_x = solver.gradient_descend(grad_method="analytical", display_process=True)
    print(f"optimal_x:{optimal_x}")
    print(f"minimum function value {obj_func(optimal_x)}")