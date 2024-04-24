import numpy as np


def sigmoid(X):
    """シグモイド関数

    $Y = \frac{1}{1 + \exp(-X)}$

    Args:
        X: k * n_Y 行列

    Returns:
        Y = k * n_Y 行列
    """
    Y = 1 / (1 + np.exp(-X))
    return Y


def sigmoid_dash(Y):
    """シグモイド関数の導関数

    シグモイド関数の導関数

    Args:
        Y: k * n_Y 行列

    Returns:
        Y_dash: k * n_Y 行列
    """

    Y_dash = Y * (1 - Y)
    return Y_dash


def relu(X):
    """ReLU

    ReLU

    Args:
        X: k * n_Y 行列
    Returns:
        Y: k * n_Y 行列
    """
    Y = np.where(X <= 0, 0, X)
    return Y


def relu_dash(Y):
    """ReLUの導関数

    ReLUの導関数

    Args:
        Y: k * n_Y 行列

    Returns:
        Y_dash: k * n_Y 行列
    """

    Y_dash = np.where(Y <= 0, 0, 1)
    return Y_dash


def identity(X):
    """恒等関数

    恒等関数

    Args:
        X: k * n_Y 行列

    Returns:
        Y: k * n_Y行列
    """
    Y = X
    return Y


def softmax(X):
    """ソフトマックス関数

    ソフトマックス関数

    Args:
        X: k * n_Y 行列

    Returns:
        Y: k * n_Y行列
    """

    Y = np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=True)
    return Y


def error_dash(Y, dY):
    """損失関数のUでの導関数

    損失関数のUでの導関数

    損失関数に二乗和誤差を、出力層の活性化関数にシグモイド関数を用いる場合と
    損失関数に交差エントロピー誤差を、出力層の活性化関数にソフトマックス関数を用いる場合に使える

    Args:
        Y: 出力層での出力値、k * n_Y 行列
        dY: 正解値、k * n_Y 行列

    Returns:
        Y_dash: k * n_Y 行列
    """
    Y_dash = Y - dY
    return Y_dash


class MiddleLayer:
    def __init__(
        self,
        n_X: int,
        n_Y: int,
        wb_width=0.01,
        eta=0.1,
        activation_func=sigmoid,
        activation_func_dash=sigmoid_dash,
    ) -> None:
        """初期化

        初期化

        Args:
            n_X(int): 入力層の次元
            n_Y(int): 出力層の次元
            wb_width(float): 重みとバイアスの広がり具合
            eta(float): 学習係数
            activation_func: 活性化関数
            activation_func_dash: 活性化関数の導関数

        Returns:
            None
        """
        # 重み
        self.W = wb_width * np.random.randn(n_X, n_Y)  # n_X * n_Y

        # バイアス
        self.b = wb_width * np.random.randn(n_Y)  # n_Y

        # 学習係数
        self.eta = eta

        # 活性化関数
        self.activation_func = activation_func
        self.activation_func_dash = activation_func_dash

    def forward(self, X):
        """順伝播

        順伝播

        Args:
            X: 入力値、k * n_X 行列

        Returns:
            Y: 出力値、k * n_Y 行列

        """

        self.X = X  # k * n_X

        U = (X @ self.W) + self.b  # k * n_Y (b はブロードキャストされる)
        self.Y = self.activation_func(U)
        # self.Y = 1 / (1 + np.exp(-U))
        # self.Y = np.maximum(u, 0)

        return self.Y

    def backward(self, dY):
        """逆伝播

        逆伝播

        Args:
            dY: 出力値の勾配、k * n_Y 行列
        """
        Y_dash = self.activation_func_dash(self.Y)
        D = dY * Y_dash
        # D = dY * (1 - self.Y) * self.Y

        dW = self.X.T @ D  # n_X * n_Y
        db = np.sum(D, axis=0)  # n_Y

        dX = D @ self.W.T  # k * n_X

        self.W -= self.eta * dW
        self.b -= self.eta * db

        return dX


class OutputLayer:
    def __init__(
        self,
        n_X: int,
        n_Y: int,
        wb_width=0.01,
        eta=0.1,
        activation_func=identity,
        error_func_dash=error_dash,
    ) -> None:
        """初期化

        初期化

        Args:
            n_X(int): 入力層の次元
            n_Y(int): 出力層の次元
            wb_width(float): 重みとバイアスの広がり具合
            eta(float): 学習係数

        Returns:
            None
        """
        # 重み
        self.W = wb_width * np.random.randn(n_X, n_Y)  # n_X * n_Y

        # バイアス
        self.b = wb_width * np.random.randn(n_Y)  # n_Y

        # 学習係数
        self.eta = eta

        # 活性化関数
        self.activation_func = activation_func
        self.error_func_dash = error_func_dash

    def forward(self, X):
        """順伝播

        順伝播

        Args:
            X: 入力値、k * n_X 行列

        Returns:
            Y: 出力値、k * n_Y 行列

        """

        self.X = X  # k * n_X

        U = (X @ self.W) + self.b  # k * n_Y (k はブロードキャストされる)
        self.Y = self.activation_func(U)

        return self.Y

    def backward(self, dY):
        """逆伝播

        逆伝播

        Args:
            dY: 正解値、k * n_Y 行列
        """

        D = self.error_func_dash(self.Y, dY)

        dW = self.X.T @ D  # n_X * n_Y
        db = np.sum(D, axis=0)  # n_Y

        dX = D @ self.W.T  # k * n_X

        self.W -= self.eta * dW
        self.b -= self.eta * db

        return dX
