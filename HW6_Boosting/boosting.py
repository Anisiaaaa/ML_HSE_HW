from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

from tqdm.auto import tqdm

from sklearn.base import ClassifierMixin


class Boosting(ClassifierMixin):

    def __init__(
        self,
        base_model_class = DecisionTreeRegressor,
        base_model_params: dict | None = None,
        n_estimators: int = 20,
        learning_rate: float = 0.05,
        random_state: int | None = None,
        verbose: bool = True,
        early_stopping_rounds: int | None = 0,
        eval_metric: str | None = None,
        cat_features: Iterable | None = None,
        cat_order: bool = False,
        subsample: float = 1.0,
        bagging_temperature: float = 1.0,
        bootstrap_type: str | None = "Bernoulli",
        rsm: float = 1.0,
        goss: bool = False,
        goss_k: float = 0.2,
        quantization_type: str | None = None,
        nbins: int = 255,
        dart: bool = False,
        dropout_rate: float = 0.05,
        loss: str = "BCE",
        focal_gamma: float = 2.0,
    ):
        super().__init__()

        self.base_model_class = base_model_class
        self.base_model_params = {} if base_model_params is None else base_model_params

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

        self.models = [0] * (n_estimators)
        self.gammas = [0] * (n_estimators)

        self.random_state = random_state  # не забудьте вставить его везде, где у вас возникает рандом
        self.verbose = verbose

        self.history = defaultdict(list)  # {"train_roc_auc": [], "train_loss": [], ...}

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))

        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric

        self.cat_features = [] if cat_features is None else list(cat_features)
        self._cat_counters = {}
        self.cat_order = cat_order

        self.subsample = subsample
        self.bagging_temperature = bagging_temperature
        self.bootstrap_type = bootstrap_type

        self.rsm = rsm
        self.features = [None] * n_estimators

        self.goss = goss
        self.goss_k = goss_k

        self.quantization_type = quantization_type
        self.nbins = nbins
        self._quantization_bins = []

        self.dart = dart
        self.dropout_rate = dropout_rate

        self.loss = loss
        self.focal_gamma = focal_gamma

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        cur_pred = np.zeros(X.shape[0])
        dropped_idx = self._drop_trees()

        for i, (model, gamma, features) in enumerate(zip(self.models, self.gammas, self.features)):
            if model != 0 and i not in dropped_idx:
                cur_pred += self.learning_rate * gamma * model.predict(X[:, features])
        
        anti_grad = self.grad_fn(y, cur_pred)
        self.base_model_params["random_state"] = self.random_state
        model = self.base_model_class(**self.base_model_params)

        features = self._select_features(X)
        X_rsm = X[:, features]
        X_fit, anti_grad_fit, sample_w = self._bootstrap(X_rsm, anti_grad)
        if sample_w is None:
            model.fit(X_fit, anti_grad_fit)
        else:
            model.fit(X_fit, anti_grad_fit, sample_weight=sample_w)
        pred_improved = model.predict(X_rsm)
        gamma = self._find_optimal_gamma(y,cur_pred, pred_improved)

        if self.dart and len(dropped_idx) > 0:
            k = len(dropped_idx)
            gamma = gamma / k
            for i in dropped_idx:
                self.gammas[i] *= k / (k + 1)

        idx = self.models.index(0)
        self.models[idx] = model
        self.gammas[idx] = gamma
        self.features[idx] = features

    def fit(
            self, X_train: np.ndarray, y_train: np.ndarray,
            eval_set: tuple[np.ndarray, np.ndarray] | None = None,
            use_best_model: bool = False,) -> None:

        self.classes_ = np.unique(y_train)  # не рекомендуется убирать, нужно для калибровки
        estimator_range = range(self.n_estimators)
        if self.verbose:
            estimator_range = tqdm(estimator_range)
        
        if self.random_state is not None:
            np.random.seed(self.random_state)

        best_iter = 0
        best_score = None
        not_improved = 0

        if self.cat_features:
            self._cat_fit(X_train, y_train)
            if self.cat_order:
                X_train = self._cat_transform_ordered(X_train, y_train)
            else:
                X_train = self._cat_transform(X_train)

            if eval_set is not None:
                X_valid, y_valid = eval_set
                X_valid = self._cat_transform(X_valid)
                eval_set = (X_valid, y_valid)
        
        if self.quantization_type is not None:
            self._fit_quantization(X_train, y_train)
            X_train = self._quantize(X_train)

            if eval_set is not None:
                X_valid, y_valid = eval_set
                X_valid = self._quantize(X_valid)
                eval_set = (X_valid, y_valid)

        for _ in estimator_range:
            self.partial_fit(X_train, y_train)
            train_predictions = np.zeros(X_train.shape[0])

            for model, gamma, features in zip(self.models, self.gammas, self.features):
                if model != 0:
                    train_predictions += self.learning_rate * gamma * model.predict(X_train[:, features])
            
            self.history["train_loss"].append(self.loss_fn(y_train, train_predictions))
            self.history["train_roc_auc"].append(roc_auc_score(y_train == 1, self.sigmoid(train_predictions)))

            if eval_set is not None:
                X_valid, y_valid = eval_set
                valid_predictions = np.zeros(X_valid.shape[0])
                for model, gamma, features in zip(self.models, self.gammas, self.features):
                    if model != 0:
                        valid_predictions += self.learning_rate * gamma * model.predict(X_valid[:, features])
                self.history["valid_loss"].append(self.loss_fn(y_valid, valid_predictions))
                self.history["valid_roc_auc"].append(roc_auc_score(y_valid == 1, self.sigmoid(valid_predictions)))

            if self.early_stopping_rounds:
                cur_score = self.history[self.eval_metric][-1]
                if best_score is None:
                    best_score = cur_score
                    best_iter = len(self.history[self.eval_metric]) - 1
                else:
                    ok = ("loss" in self.eval_metric and cur_score < best_score) or (
                        "loss" not in self.eval_metric and cur_score > best_score
                    )
                    if ok:
                        best_score = cur_score
                        best_iter = len(self.history[self.eval_metric]) - 1
                        not_improved = 0
                    else:
                        not_improved += 1

                    if not_improved >= self.early_stopping_rounds:
                        break
        
        if use_best_model and self.early_stopping_rounds:
            for i in range(best_iter + 1, self.n_estimators):
                self.models[i] = 0
                self.gammas[i] = 0

        for key in self.history:
            self.history[key] = np.array(self.history[key])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.cat_features:
            X = self._cat_transform(X)
        
        if self.quantization_type is not None:
            X = self._quantize(X)

        preds = np.zeros(X.shape[0])
        for model, gamma, features in zip(self.models, self.gammas, self.features):
            if model != 0:
                preds += self.learning_rate * gamma * model.predict(X[:, features])
        probs = self.sigmoid(preds)
        return np.column_stack([1 - probs, probs])

    def _find_optimal_gamma(
        self,
        y: np.ndarray,
        old_predictions: np.ndarray, 
        new_predictions: np.ndarray
    ) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [
            self.loss_fn(y, old_predictions + gamma * new_predictions)
            for gamma in gammas
        ]
        return gammas[np.argmin(losses)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return roc_auc_score(y == 1, self.predict_proba(X)[:, 1])
    
    def plot_history(self, keys):
        import matplotlib.pyplot as plt

        if isinstance(keys, str):
            keys = [keys]
        for key in keys:
            plt.plot(self.history[key], label=key)
            
        plt.xlabel("Iteration")
        plt.ylabel("Value")
        plt.legend()
        plt.grid()
        plt.show()
    
    def _cat_fit(self, X: np.array, y: np.ndarray) -> None:
        self._global_mean = (y == 1).mean()
        for col in self.cat_features:
            self._cat_counters[col] = {}
            for cat in np.unique(X[:, col]):
                y_target = y[X[:, col] == cat]
                self._cat_counters[col][cat] = (y_target == 1).mean()
    
    def _cat_transform(self, X: np.ndarray) -> np.ndarray:
        X_transform = X.copy()
        for col in self.cat_features:
            target_map = self._cat_counters[col]
            X_transform[:, col] = [target_map.get(x, self._global_mean) for x in X[:, col]]
        return X_transform.astype(float)
    
    def _cat_transform_ordered(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        X_transform = X.copy()
        mean = (y == 1).mean()

        for col in self.cat_features:
            y1 = {}
            cnts = {}
            for i in range(X.shape[0]):
                cat = X[i, col]
                if cat in cnts:
                    X_transform[i, col] = y1[cat] / cnts[cat]
                else:
                    X_transform[i, col] = mean
                y1[cat] = y1.get(cat, 0) + (y[i] == 1)
                cnts[cat] = cnts.get(cat, 0) + 1

        return X_transform.astype(float)

    def _bootstrap(self, X: np.ndarray, y: np.ndarray):
        if self.goss:
            n = X.shape[0]
            n_big = int(self.goss_k * n)

            sorted_idx = np.argsort(np.abs(y))[::-1]
            big_idx = sorted_idx[:n_big]
            small_idx = sorted_idx[n_big:]
            small_mask = np.random.random(len(small_idx)) < self.subsample
            sampled_small_idx = small_idx[small_mask]

            idx = np.concatenate([big_idx, sampled_small_idx])
            w = np.ones(len(idx))
            w[n_big:] = (1 - self.goss_k) / self.subsample

            return X[idx], y[idx], w
    
        if self.bootstrap_type == "Bernoulli":
            mask = np.random.random(X.shape[0]) < self.subsample
            return X[mask], y[mask], None

        if self.bootstrap_type == "Bayesian":
            w = (-np.log(np.random.random(X.shape[0]))) ** self.bagging_temperature
            return X, y, w
    
    def _select_features(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[1]
        mask = np.random.random(n) < self.rsm
        return np.where(mask)[0]
    
    def _fit_quantization(self, X: np.ndarray, y: np.ndarray) -> None:
        for col in range(X.shape[1]):
            values = X[:, col].astype(float)
            if self.quantization_type == "uniform":
                bins = np.linspace(values.min(), values.max(), self.nbins + 1)[1:-1]
            elif self.quantization_type == "quantile":
                bins = np.quantile(values, np.linspace(0, 1, self.nbins + 1))[1:-1]
            elif self.quantization_type == "piecewise":
                tree = DecisionTreeClassifier(max_leaf_nodes=self.nbins,random_state=self.random_state)
                tree.fit(values.reshape(-1, 1), y)
                thresholds = tree.tree_.threshold
                bins = thresholds[thresholds != -2]
                bins = np.sort(bins)
            self._quantization_bins.append(bins)
    
    def _quantize(self, X: np.ndarray) -> np.ndarray:
        X_q = np.zeros(X.shape, dtype=int)
        for col in range(X.shape[1]):
            values = X[:, col].astype(float)
            X_q[:, col] = np.digitize(values, self._quantization_bins[col])
        return X_q
    
    def get_feature_importance(self, X: np.ndarray | None = None, y: np.ndarray | None = None, type: str = "split") -> np.ndarray:
        if X is None:
            raise ValueError("X cannot be None")
        
        if self.cat_features:
            X = self._cat_transform(X)
        if self.quantization_type is not None:
            X = self._quantize(X)
        importance = np.zeros(X.shape[1])

        if type == "split":
            for model, gamma, features in zip(self.models, self.gammas, self.features):
                if model == 0:
                    continue
                tree_imp = model.feature_importances_
                importance[features] += gamma * tree_imp
        
        if type == "gain":
            y = np.asarray(y)
            pred = np.zeros(X.shape[0])

            for model, gamma, features in zip(self.models, self.gammas, self.features):
                if model == 0:
                    continue
                anti_grad = self.grad_fn(y, pred)
                path = model.decision_path(X[:, features])
                tree = model.tree_

                for i in range(X.shape[0]):
                    start = path.indptr[i]
                    end = path.indptr[i + 1]
                    nodes = path.indices[start:end]

                    for node in nodes:
                        local_feature = tree.feature[node]
                        if local_feature == -2:
                            continue
                        orig_feature = features[local_feature]
                        node_value = tree.value[node].ravel()[0]
                        importance[orig_feature] += abs(gamma * anti_grad[i] * node_value)
                pred += self.learning_rate * gamma * model.predict(X[:, features])
            
        importance = importance / importance.sum()
        return importance
    
    def _drop_trees(self) -> list[int]:
        trained_idx = [i for i, model in enumerate(self.models) if model != 0]
        if not self.dart or len(trained_idx) == 0:
            return []
        mask = np.random.random(len(trained_idx)) < self.dropout_rate
        dropped_idx = list(np.array(trained_idx)[mask])
        return dropped_idx
    
    def loss_fn(self, y: np.ndarray, z: np.ndarray) -> float:
        p = self.sigmoid(y * z)
        if self.loss == "BCE":
            return -np.log(p).mean()
        if self.loss == "Focal":
            gamma = self.focal_gamma
            return -(((1 - p) ** gamma) * np.log(p)).mean()

    def grad_fn(self, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        p = self.sigmoid(y * z)
        if self.loss == "BCE":
            return y * (1 - p)
        if self.loss == "Focal":
            gamma = self.focal_gamma
            return y * ((1 - p) ** gamma) * ((1 - p) - gamma * p * np.log(p))