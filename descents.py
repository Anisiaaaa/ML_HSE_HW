import numpy as np
from abc import ABC, abstractmethod
from interfaces import LearningRateSchedule, AbstractOptimizer, LinearRegressionInterface


# ===== Learning Rate Schedules =====
class ConstantLR(LearningRateSchedule):
    def __init__(self, lr: float):
        self.lr = lr

    def get_lr(self, iteration: int) -> float:
        return self.lr


class TimeDecayLR(LearningRateSchedule):
    def __init__(self, lambda_: float = 1.0):
        self.s0 = 1
        self.p = 0.5
        self.lambda_ = lambda_

    def get_lr(self, iteration: int) -> float:
        """
        returns: float, learning rate для iteration шага обучения
        """
        return self.lambda_ * (self.s0 / (self.s0 + iteration)) ** self.p


# ===== Base Optimizer =====
class BaseDescent(AbstractOptimizer, ABC):
    """
    Оптимизатор, имплементирующий градиентный спуск.
    Ответственен только за имплементацию общего алгоритма спуска.
    Все его составные части (learning rate, loss function+regularization) находятся вне зоны ответственности этого класса (см. Single Responsibility Principle).
    """
    def __init__(self, 
                 lr_schedule: LearningRateSchedule = TimeDecayLR(), 
                 tolerance: float = 1e-6,
                 max_iter: int = 1000
                ):
        self.lr_schedule = lr_schedule
        self.tolerance = tolerance
        self.max_iter = max_iter

        self.iteration = 0
        self.model: LinearRegressionInterface = None

    @abstractmethod
    def _update_weights(self) -> np.ndarray:
        """
        Вычисляет обновление согласно конкретному алгоритму и обновляет веса модели, перезаписывая её атрибут.
        Не имеет прямого доступа к вычислению градиента в точке, для подсчета вызывает model.compute_gradients.

        returns: np.ndarray, w_{k+1} - w_k
        """
        pass

    def _step(self) -> np.ndarray:
        """
        Проводит один полный шаг интеративного алгоритма градиентного спуска

        returns: np.ndarray, w_{k+1} - w_k
        """
        delta = self._update_weights()
        self.iteration += 1
        return delta

    def optimize(self) -> None:
        """
        Оркестрирует весь алгоритм градиентного спуска.
        """
        ...
        self.iteration = 0
        self.model.loss_history = []
        self.model.loss_history.append(self.model.compute_loss())

        while self.iteration < self.max_iter:
            d_w = self._update_weights()
            self.model.loss_history.append(self.model.compute_loss())
            if np.isnan(d_w).any():
                break
            if np.sum(d_w ** 2) < self.tolerance:
                break
            self.iteration += 1


# ===== Specific Optimizers =====
class VanillaGradientDescent(BaseDescent):
    def _update_weights(self) -> np.ndarray:
        lr = self.lr_schedule.get_lr(self.iteration)
        X_train = self.model.X_train
        y_train = self.model.y_train
        gradient = self.model.compute_gradients(X_train, y_train)
        delt_w = -lr * gradient
        self.model.w = self.model.w + delt_w
        return delt_w


class StochasticGradientDescent(BaseDescent):
    def __init__(self, *args, batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size

    def _update_weights(self) -> np.ndarray:
        lr = self.lr_schedule.get_lr(self.iteration)
        n = self.model.X_train.shape[0]
        batch = np.random.randint(0, n, size=self.batch_size)
        X_batch = self.model.X_train[batch]
        y_batch = self.model.y_train[batch]
        grad = self.model.compute_gradients(X_batch, y_batch)
        delt_w = -lr * grad
        self.model.w += delt_w
        return delt_w
    


class SAGDescent(BaseDescent):
    def __init__(self, *args, batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_memory = None
        self.grad_sum = None
        self.batch_size = batch_size

    def _update_weights(self) -> np.ndarray:
        X_train = self.model.X_train
        y_train = self.model.y_train
        num_objects, num_features = X_train.shape
        lr = self.lr_schedule.get_lr(self.iteration)

        if self.grad_memory is None:
            self.grad_memory = np.zeros((num_objects, num_features))
            self.grad_sum = np.zeros(num_features)

        batch = np.random.randint(0, num_objects, size=self.batch_size)
        for i in batch:
            grad_new = self.model.compute_gradients(X_train[i:i+1], y_train[i:i+1])
            grad_old = self.grad_memory[i].copy()
            self.grad_memory[i] = grad_new
            self.grad_sum += (grad_new - grad_old)

        grad_avg = self.grad_sum / num_objects
        delt_w = -lr * grad_avg
        self.model.w += delt_w
        return delt_w

class MomentumDescent(BaseDescent):
    def __init__(self,  *args, beta=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.velocity = None

    def _update_weights(self) -> np.ndarray:
        X_train = self.model.X_train
        y_train = self.model.y_train
        lr = self.lr_schedule.get_lr(self.iteration)

        if self.velocity is None:
            self.velocity = np.zeros(self.model.w.shape)
        
        grad = self.model.compute_gradients(X_train, y_train)
        self.velocity = self.beta * self.velocity + lr * grad
        delt_w = -self.velocity
        self.model.w += delt_w
        return delt_w


class Adam(BaseDescent):
    def __init__(self, *args, beta1=0.9, beta2=0.999, eps=1e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None

    def _update_weights(self) -> np.ndarray:
        X_train = self.model.X_train
        y_train = self.model.y_train
        lr = self.lr_schedule.get_lr(self.iteration)
        grad = self.model.compute_gradients(X_train, y_train)

        if self.m is None:
            self.m = np.zeros(self.model.w.shape)
            self.v = np.zeros(self.model.w.shape)
        
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        i = self.iteration + 1
        m = self.m / (1 - self.beta1 ** i)
        v = self.v / (1 - self.beta2 ** i)

        delt_w = -lr * m / (np.sqrt(v) + self.eps)
        self.model.w += delt_w
        return delt_w


# ===== Non-iterative Algorithms ====
class AnalyticSolutionOptimizer(AbstractOptimizer):
    """
    Универсальный дамми-класс для вызова аналитических решений 
    """
    def __init__(self):
        self.model = None
    

    def optimize(self) -> None:
        """
        Определяет аналитическое решение и назначает его весам модели.
        """
        # не должна содержать непосредственных формул аналитического решения, за него ответственен другой объект
        ...
        X = self.model.X_train
        y = self.model.y_train

        w = self.model.loss_function.analytic_solution(X, y)
        self.model.w = w

