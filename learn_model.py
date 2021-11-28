import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import dill


class Node:

    def __init__(self, index, t, true_branch, false_branch):
        self.index = index  # индекс признака, по которому ведется сравнение с порогом в этом узле
        self.t = t  # значение порога
        self.true_branch = true_branch  # поддерево, удовлетворяющее условию в узле
        self.false_branch = false_branch  # поддерево, не удовлетворяющее условию в узле


class BaseTree:

    def __init__(self, bootstrap,
                 max_depth,
                 max_leaf_nodes,
                 min_leaf_samples,
                 leaf_class):

        self.bootstrap = bootstrap
        self.max_depth = max_depth
        self.nodes = []
        self.leaves = []
        self.depth = 0
        self.max_leaves = max_leaf_nodes
        self.min_objects = min_leaf_samples
        self.tree = None
        self.Leaf = leaf_class

    # Разбиение датасета в узле
    def split(self,
              data,
              labels,
              column_index,
              t):

        left = np.where(data[:, column_index] <= t)
        right = np.where(data[:, column_index] > t)

        true_data = data[left]
        false_data = data[right]

        true_labels = labels[left]
        false_labels = labels[right]

        return true_data, false_data, true_labels, false_labels

    # Расчет прироста
    def gain(self,
             left_labels,
             right_labels,
             root):

        # доля выборки, ушедшая в левое поддерево
        p = float(left_labels.shape[0]) / (left_labels.shape[0] + right_labels.shape[0])

        return root - p * self.criterion(left_labels) - (1 - p) * self.criterion(right_labels)

        # Нахождение наилучшего разбиения

    def find_best_split(self, data, labels):

        #  обозначим минимальное количество объектов в узле
        min_samples_leaf = 5

        root = self.criterion(labels)

        best_gain = 0
        best_t = None
        best_index = None

        n_features = data.shape[1]

        for index in self.bootstrap:
            # будем проверять только уникальные значения признака, исключая повторения
            t_values = np.unique(data[:, index])

            for t in t_values:
                true_data, false_data, true_labels, false_labels = self.split(data, labels, index, t)
                #  пропускаем разбиения, в которых в узле остается менее 5 объектов
                if len(true_data) < min_samples_leaf or len(false_data) < min_samples_leaf:
                    continue

                current_gain = self.gain(true_labels, false_labels, root)

                #  выбираем порог, на котором получается максимальный прирост качества
                if current_gain > best_gain:
                    best_gain, best_t, best_index = current_gain, t, index

        return best_gain, best_t, best_index

    # Построение дерева с помощью рекурсивной функции
    def build_tree(self,
                   data,
                   labels):

        gain, t, index = self.find_best_split(data, labels)

        #  Базовый случай 2 - прекращаем рекурсию, когда достигли максимальной глубины дерева
        if self.depth > self.max_depth:
            self.leaves.append(self.Leaf(data, labels))
            return self.Leaf(data, labels)

        #  Базовый случай 3 - прекращаем рекурсию, когда достигли максимального количества листьев
        if len(self.leaves) >= self.max_leaves - 1 or self.depth >= self.max_leaves - 1:
            self.leaves.append(self.Leaf(data, labels))
            return self.Leaf(data, labels)

        #  Базовый случай 4 - прекращаем рекурсию, когда достигли минимального количества объектов в листе
        if len(data) <= self.min_objects:
            self.leaves.append(self.Leaf(data, labels))
            return self.Leaf(data, labels)

        #  Базовый случай 1 - прекращаем рекурсию, когда нет прироста в качества
        if gain == 0:
            self.leaves.append(self.Leaf(data, labels))
            return self.Leaf(data, labels)

        self.depth += 1

        true_data, false_data, true_labels, false_labels = self.split(data, labels, index, t)

        # Рекурсивно строим два поддерева
        true_branch = self.build_tree(true_data, true_labels)
        false_branch = self.build_tree(false_data, false_labels)

        # Возвращаем класс узла со всеми поддеревьями, то есть целого дерева
        self.nodes.append(Node(index, t, true_branch, false_branch))
        return Node(index, t, true_branch, false_branch)

    def predict_object(self,
                       obj,
                       node):

        #  Останавливаем рекурсию, если достигли листа
        if isinstance(node, self.Leaf):
            answer = node.prediction
            return answer
        if obj[node.index] <= node.t:
            return self.predict_object(obj, node.true_branch)
        else:
            return self.predict_object(obj, node.false_branch)

    def fit(self, data, labels):
        self.tree = self.build_tree(data, labels)
        return self

    def predict(self, data):

        classes = []
        for obj in data:
            prediction = self.predict_object(obj, self.tree)
            classes.append(prediction)
        return classes


class Leaf_regr:

    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
        self.prediction = self.predict()

    def predict(self):
        return self.targets.mean()


class Regression_Tree(BaseTree):
    def __init__(self, bootstrap=[1, 2, 3, 4],
                 max_depth=np.inf,
                 max_leaf_nodes=np.inf,
                 min_leaf_samples=1,
                 leaf_class=Leaf_regr):
        super().__init__(bootstrap=bootstrap,
                         max_depth=max_depth,
                         max_leaf_nodes=max_leaf_nodes,
                         min_leaf_samples=min_leaf_samples,
                         leaf_class=Leaf_regr)

    # Расчет дисперсии
    def criterion(self, targets):
        return np.mean((targets - targets.mean()) ** 2)


def gb_predict(X, trees_list, eta):
    predictions = np.array(
        [sum([eta * alg.predict([x])[0] for alg in trees_list]) for x in X]
    )

    return predictions


def mean_squared_error(y_real, prediction):
    return (sum((y_real - prediction) ** 2)) / len(y_real)


def residual(y, z):
    return - (z - y)


def gb_fit(n_trees, max_depth, X_train, X_test, y_train, y_test, eta):
    # Деревья будем записывать в список
    trees = []

    # Будем записывать ошибки на обучающей и тестовой выборке на каждой итерации в список
    train_errors = []
    test_errors = []

    np.random.seed(1200)
    bootstrap = [1, 2, 3, 4, 5]

    for i in range(n_trees):
        bootstrap_copy = bootstrap.copy()

        r1 = np.random.randint(1, 6)
        if r1 != 4:
            bootstrap_copy.remove(r1)

        r2 = np.random.randint(1, 6)
        if r2 != 4 and r2 != r1:
            bootstrap_copy.remove(r2)

        f1 = np.random.randint(6)
        if f1 != 0:
            bootstrap_copy.append(5 + f1)

        f2 = np.random.randint(6)
        if f2 != 0 and f2 != f1:
            bootstrap_copy.append(5 + f2)

        f3 = np.random.randint(6)
        if f3 != 0 and f3 != f1 and f3 != f2:
            bootstrap_copy.append(5 + f3)

        tree = Regression_Tree(max_depth=max_depth, bootstrap=bootstrap_copy)

        # первый алгоритм просто обучаем на выборке и добавляем в список
        if len(trees) == 0:
            # обучаем первое дерево на обучающей выборке
            tree.fit(X_train, y_train)

            train_errors.append(mean_squared_error(y_train, gb_predict(X_train, trees, eta)))
            test_errors.append(mean_squared_error(y_test, gb_predict(X_test, trees, eta)))
        else:
            # Получим ответы на текущей композиции
            target = gb_predict(X_train, trees, eta)

            # алгоритмы начиная со второго обучаем на сдвиг
            resid = residual(y_train, target)
            tree.fit(X_train, resid)

            train_errors.append(mean_squared_error(y_train, gb_predict(X_train, trees, eta)))
            test_errors.append(mean_squared_error(y_test, gb_predict(X_test, trees, eta)))

        trees.append(tree)

    return trees, train_errors, test_errors

if __name__ == "__main__":
    TRAIN_DATASET_PATH = "gb-tutors-expected-math-exam-results/train.csv"
    train_df = pd.read_csv(TRAIN_DATASET_PATH)
    targets = train_df['mean_exam_points'].to_numpy()
    data = train_df.copy().drop(columns=['mean_exam_points'], axis=1).to_numpy()

    train_data_main, test_data_main, train_target_main, test_target_main = train_test_split(data, targets,
                                                                                            test_size=0.32,
                                                                                            random_state=34)

    trees, train_errors, test_errors = gb_fit(48, 4, train_data_main, test_data_main,
                                              train_target_main, test_target_main, 0.1516)

    with open('model.dill', 'wb') as f:
        dill.dump(trees, f)



