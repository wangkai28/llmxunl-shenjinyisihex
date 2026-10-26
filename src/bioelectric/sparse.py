from __future__ import annotations

import numpy as np


class SparseMatrix:
    """轻量 COO 稀疏矩阵（numpy 数组存储，无 scipy 依赖）。

    热路径（dot / coo_arrays / set_coo）全部用向量化 numpy 操作；
    标量读写（测试用）用线性查找，接受较慢。
    """

    def __init__(self, n_rows: int, n_cols: int):
        self.shape = (n_rows, n_cols)
        self._rows = np.empty(0, dtype=np.int64)
        self._cols = np.empty(0, dtype=np.int64)
        self._data = np.empty(0, dtype=np.float64)

    @property
    def nnz(self) -> int:
        return int(self._data.size)

    def copy(self):
        m = SparseMatrix(self.shape[0], self.shape[1])
        m._rows = self._rows.copy()
        m._cols = self._cols.copy()
        m._data = self._data.copy()
        return m

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            i, j = key
            if isinstance(i, int) and isinstance(j, int):
                return self._get_scalar(i, j)
            return self.to_dense()[key]
        return self.to_dense()[key]

    def __setitem__(self, key, value):
        if isinstance(key, tuple) and len(key) == 2:
            i, j = key
            if isinstance(i, int) and isinstance(j, int):
                self._set_scalar(i, j, value)
                return
        raise NotImplementedError("only scalar (i, j) assignment is supported")

    def _get_scalar(self, i: int, j: int) -> float:
        mask = (self._rows == i) & (self._cols == j)
        idx = np.flatnonzero(mask)
        return float(self._data[idx[0]]) if idx.size else 0.0

    def _set_scalar(self, i: int, j: int, value: float):
        mask = (self._rows == i) & (self._cols == j)
        idx = np.flatnonzero(mask)
        if idx.size:
            if value == 0.0:
                keep = ~mask
                self._rows = self._rows[keep]
                self._cols = self._cols[keep]
                self._data = self._data[keep]
            else:
                self._data[idx[0]] = value
        elif value != 0.0:
            self._rows = np.append(self._rows, i)
            self._cols = np.append(self._cols, j)
            self._data = np.append(self._data, value)

    def to_dense(self) -> np.ndarray:
        out = np.zeros(self.shape, dtype=float)
        if self.nnz:
            out[self._rows, self._cols] = self._data
        return out

    def dot(self, vector: np.ndarray) -> np.ndarray:
        """矩阵乘向量：结果[i] = sum_j W[i,j] * vector[j]（COO 按行累加）。"""
        if not self.nnz:
            return np.zeros(self.shape[0], dtype=float)
        contributions = self._data * vector[self._cols]
        return np.bincount(self._rows, weights=contributions, minlength=self.shape[0])

    def coo_arrays(self):
        return self._rows, self._cols, self._data

    def set_coo(self, rows, cols, data):
        self._rows = np.asarray(rows, dtype=np.int64)
        self._cols = np.asarray(cols, dtype=np.int64)
        self._data = np.asarray(data, dtype=np.float64)
