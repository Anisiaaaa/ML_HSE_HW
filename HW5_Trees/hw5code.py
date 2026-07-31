import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов нужно брать среднее двух соседних при сортировке значений признака
    * Поведение функции в случае константного признака может быть любым
    * При одинаковых приростах критерия Джини для нескольких порогов нужно выбирать сплит, у которого значение порога минимально
    * Достаточно поддерживать только бинарную классификацию.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов, len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно разделить на две различные подвыборки или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds, len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    
    sort_ind = np.argsort(feature_vector)
    feature_sort = feature_vector[sort_ind]
    target_sort = target_vector[sort_ind]
    thresholds = (feature_sort[1:] + feature_sort[:-1]) / 2

    n = len(target_sort)
    left_n = np.arange(1, n)
    right_n = n - left_n

    left_class1 = np.cumsum(target_sort)[:-1]
    right_class1 = np.sum(target_sort) - left_class1

    p1_left = left_class1 / left_n
    p0_left = 1 - p1_left
    p1_right = right_class1 / right_n
    p0_right = 1 - p1_right

    H_left = 1 - p0_left**2 - p1_left**2
    H_right = 1 - p0_right**2 - p1_right**2

    ginis = - (left_n / n) * H_left - (right_n / n) * H_right

    unique = feature_sort[1:] != feature_sort[:-1]
    thresholds = thresholds[unique]
    ginis = ginis[unique]

    best_ind = np.argmax(ginis)
    best_gini = ginis[best_ind]
    best_thresholds = thresholds[best_ind]

    return thresholds, ginis, best_thresholds, best_gini



class DecisionTree:
    """
    Простое классификационное дерево, поддерживающее:
    * real / categorical признаки
    * binary цели (метки могут быть числами или строками)
    * ограничения max_depth, min_samples_split, min_samples_leaf (как в sklearn по смыслу)

    ВНИМАНИЕ: в методе _fit_node ниже могут быть намеренно оставлены некоторые ошибки.
    Их нужно исправить в рамках задания.
    """
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return
        
        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if self._min_samples_split is not None and len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(0, sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count #исправила
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError

            if len(np.unique(feature_vector)) == 1:
                continue

            thresholds, ginis, _, _ = find_best_split(feature_vector, sub_y)

            if self._min_samples_leaf is not None:
                left_sizes = np.array([np.sum(feature_vector < t) for t in thresholds])
                right_sizes = len(feature_vector) - left_sizes
                ok = (left_sizes >= self._min_samples_leaf) & (right_sizes >= self._min_samples_leaf)

                if not np.any(ok):
                    continue

                thresholds = thresholds[ok]
                ginis = ginis[ok]

            best_ind = np.argmax(ginis)
            threshold = thresholds[best_ind]
            gini = ginis[best_ind]

            cur_split = feature_vector <= threshold

            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = cur_split
                
                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]
        else:
            feature = node["feature_split"]
            if self._feature_types[feature] == "real":
                if x[feature] < node["threshold"]:
                    return self._predict_node(x, node["left_child"])
                else:
                    return self._predict_node(x, node["right_child"])
            else:
                if x[feature] in node["categories_split"]:
                    return self._predict_node(x, node["left_child"])
                else:
                    return self._predict_node(x, node["right_child"])



    def fit(self, X, y):
        self.classes = np.unique(y)
        y = (y == self.classes[1]).astype(int)
        self._fit_node(X, y, self._tree, depth = 0)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self.classes[self._predict_node(x, self._tree)])
        return np.array(predicted)
