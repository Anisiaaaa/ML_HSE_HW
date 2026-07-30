import numpy as np 
from interfaces import LossFunction, LossFunctionClosedFormMixin, LinearRegressionInterface, AbstractOptimizer
from descents import AnalyticSolutionOptimizer
from typing import Dict, Type, Optional, Callable
from abc import abstractmethod, ABC



class MSELoss(LossFunction, LossFunctionClosedFormMixin):

    def __init__(self, analytic_solution_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = None):

        if analytic_solution_func is None:
            self.analytic_solution_func = self._plain_analytic_solution
        else:
            self.analytic_solution_func = analytic_solution_func

        

    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        """
        X: np.ndarray, матрица регрессоров 
        y: np.ndarray, вектор таргета
        w: np.ndarray, вектор весов

        returns: float, значение MSE на данных X,y для весов w
        """
        Q = np.mean((X @ w - y) ** 2)
        return Q


    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        X: np.ndarray, матрица регрессоров 
        y: np.ndarray, вектор таргета
        w: np.ndarray, вектор весов

        returns: np.ndarray, численный градиент MSE в точке w
        """
        l = X.shape[0]
        grad = (2 / l) * X.T @ (X @ w - y)
        return grad

    def analytic_solution(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Возвращает решение по явной формуле (closed-form solution)

        X: np.ndarray, матрица регрессоров 
        y: np.ndarray, вектор таргета

        returns: np.ndarray, оптимальный по MSE вектор весов, вычисленный при помощи аналитического решения для данных X, y
        """
        # Функция-диспатчер в одну из истинных функций для вычисления решения по явной формуле (closed-form)
        # Необходима в связи c наличием интерфейса analytic_solution у любого лосса; 
        # self-injection даёт возможность выбирать, какое именно closed-form решение использовать
        return self.analytic_solution_func(X, y)
    
    import numpy as np

    @classmethod 
    def _plain_analytic_solution(cls, X: np.ndarray, y: np.ndarray) -> np.ndarray: 
        return np.linalg.inv(X.T @ X) @ X.T @ y


class LogCoshLoss(LossFunction):

    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        Q = np.mean(np.log(np.cosh(X @ w - y)))
        return Q

    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        grad = (1 / n) * X.T @ np.tanh(X @ w - y)
        return grad

import numpy as np


class HuberLoss(LossFunction):

    def __init__(self, delta: float = 1.0):
        self.delta = delta

    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        module = np.abs(X @ w - y)
        part_1 = 0.5 * (X @ w - y) **2
        part_2 = self.delta * module - 0.5 * self.delta**2
        Q = np.where(module <= self.delta, part_1, part_2)
        return np.mean(Q)

    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        module = np.abs(X @ w - y)
        mi = np.where(module <= self.delta, X @ w - y, self.delta * np.sign(X @ w - y))
        grad = (1 / n) * X.T @ mi
        return grad


class L2Regularization(LossFunction):

    def __init__(self, core_loss: LossFunction, mu_rate: float = 1.0,
                 analytic_solution_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = None):
        self.core_loss = core_loss
        self.mu_rate = mu_rate

        # analytic_solution_func is meant to be passed separately, 
        # as it is not linear to core solution

    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:  #вообще его тут не было, но в абстрактном классе сказано, что он должен быть
        core = self.core_loss.loss(X, y, w)
        penalty = 0.5 * self.mu_rate * np.sum(w ** 2)
        return core + penalty

    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        core_part = self.core_loss.gradient(X, y, w)
        penalty_part = self.mu_rate * w
        return core_part + penalty_part


class CustomLinearRegression(LinearRegressionInterface):
    def __init__(
        self,
        optimizer: AbstractOptimizer,
        # l2_coef: float = 0.0,
        loss_function: LossFunction = MSELoss()
    ):
        self.optimizer = optimizer
        self.optimizer.set_model(self)

        # self.l2_coef = l2_coef
        self.loss_function = loss_function
        self.loss_history = []
        self.w = None
        self.X_train = None
        self.y_train = None
        

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        returns: np.ndarray, вектор \hat{y}
        """
        return X @ self.w

    def compute_gradients(self, X_batch: np.ndarray | None = None, y_batch: np.ndarray | None = None) -> np.ndarray:
        if X_batch is None:
            X_batch = self.X_train
            y_batch = self.y_train
        
        return self.loss_function.gradient(X_batch, y_batch, self.w)


    def compute_loss(self, X_batch: np.ndarray | None = None, y_batch: np.ndarray | None = None) -> float:
        if X_batch is None:
            X_batch = self.X_train
            y_batch = self.y_train

        return self.loss_function.loss(X_batch, y_batch, self.w)


    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Инициирует обучение модели заданным функцией потерь и оптимизатором способом.
        
        X: np.ndarray, 
        y: np.ndarray
        """
        self.X_train, self.y_train = X, y
        if self.w is None:
            self.w = np.zeros(X.shape[1])
        self.optimizer.optimize()
