"""
High-level Matrix wrapper class.

Provides a Pythonic interface to the low-level C matrix operations
exposed through ctypes in _core module.
"""

from __future__ import annotations
import ctypes
from ctypes import POINTER
from typing import Optional, Union, Iterable, Any

from ._core import (
    ensure_signatures,
    CMatrix,
    CVector,
    DataType,
    MatrixStatus,
    NormType,
    _get_element_size,
    _python_to_c_array,
    _c_array_to_python,
    _get_scalar_value,
    _check_matrix_status,
    _check_vector_status,
    DimensionError,
    MathError,
    BasicError,
)


class Matrix:
    """
    High-level Matrix wrapper.

    Wraps the low-level CMatrix struct and provides Pythonic operations
    for creating, manipulating, and performing linear algebra on matrices.
    """

    def __init__(
        self,
        m: int,
        n: int,
        data_type: int = DataType.REAL,
        init_values: Optional[Iterable] = None,
    ):
        """
        Initialize a Matrix.

        Args:
            m: Number of rows
            n: Number of columns
            data_type: DataType enum (INT, REAL, or COMPLEX)
            init_values: Optional iterable of initial values (row-major order)
        """
        self.data_type = data_type
        self._c_matrix = CMatrix()

        lib = ensure_signatures()

        if init_values is not None:
            flat_values = list(init_values)
            c_array = _python_to_c_array(flat_values, data_type)
            self._c_matrix = lib.Matrix_new_vals(m, n, c_array, data_type)
        else:
            self._c_matrix = lib.Matrix_new(m, n, data_type)

    def __del__(self):
        """Cleanup: free the underlying C matrix."""
        try:
            lib = ensure_signatures()
            lib.Matrix_free(POINTER(CMatrix)(self._c_matrix))
        except:
            pass  # Ignore errors during cleanup

    @property
    def shape(self) -> tuple[int, int]:
        """Return (rows, columns) of the matrix."""
        return (self._c_matrix.m, self._c_matrix.n)

    @property
    def rows(self) -> int:
        """Return number of rows."""
        return self._c_matrix.m

    @property
    def cols(self) -> int:
        """Return number of columns."""
        return self._c_matrix.n

    @classmethod
    def zeros(cls, m: int, n: int, data_type: int = DataType.REAL) -> Matrix:
        """Create a zero matrix of shape (m, n)."""
        lib = ensure_signatures()
        mat = cls.__new__(cls)
        mat.data_type = data_type
        mat._c_matrix = lib.Matrix_zeros(m, n, data_type)
        return mat

    @classmethod
    def diag(cls, vector: Vector) -> Matrix:
        """Create a diagonal matrix from a vector."""
        lib = ensure_signatures()
        mat = cls.__new__(cls)
        mat.data_type = vector.data_type
        mat._c_matrix = lib.Matrix_diag(POINTER(CVector)(vector._c_vector))
        return mat

    @classmethod
    def random_normal(
        cls,
        m: int,
        n: int,
        mean: float = 0.0,
        variance: float = 1.0,
        data_type: int = DataType.REAL,
    ) -> Matrix:
        """Create a matrix with random values from normal distribution."""
        lib = ensure_signatures()
        mat = cls.__new__(cls)
        mat.data_type = data_type
        mat._c_matrix = lib.Matrix_new_random_normal(m, n, mean, variance, data_type)
        return mat

    @classmethod
    def random_uniform(
        cls,
        m: int,
        n: int,
        min_val: float = 0.0,
        max_val: float = 1.0,
        data_type: int = DataType.REAL,
    ) -> Matrix:
        """Create a matrix with random values from uniform distribution."""
        lib = ensure_signatures()
        mat = cls.__new__(cls)
        mat.data_type = data_type
        mat._c_matrix = lib.Matrix_new_random_uniform(m, n, min_val, max_val, data_type)
        return mat

    def copy(self) -> Matrix:
        """Return a deep copy of this matrix."""
        lib = ensure_signatures()
        target = Matrix(self.rows, self.cols, self.data_type)
        lib.Matrix_copy(
            POINTER(CMatrix)(target._c_matrix), POINTER(CMatrix)(self._c_matrix)
        )
        return target

    def get_at(self, i: int, j: int) -> Any:
        """Get element at position (i, j)."""
        lib = ensure_signatures()
        element_size = _get_element_size(self.data_type)
        buffer = ctypes.create_string_buffer(element_size)
        status = lib.Matrix_get_at(
            ctypes.cast(buffer, ctypes.c_void_p), POINTER(CMatrix)(self._c_matrix), i, j
        )
        _check_matrix_status(status)
        return _get_scalar_value(ctypes.cast(buffer, ctypes.c_void_p), self.data_type)

    def set_at(self, i: int, j: int, value: Union[int, float, complex]) -> None:
        """Set element at position (i, j)."""
        lib = ensure_signatures()
        c_value = _python_to_c_array([value], self.data_type)
        status = lib.Matrix_set_at(POINTER(CMatrix)(self._c_matrix), i, j, c_value)
        _check_matrix_status(status)

    def __getitem__(self, key: tuple[int, int]) -> Any:
        """Support matrix[i, j] indexing."""
        return self.get_at(key[0], key[1])

    def __setitem__(self, key: tuple[int, int], value: Any) -> None:
        """Support matrix[i, j] = value assignment."""
        self.set_at(key[0], key[1], value)

    def add(self, other: Matrix) -> Matrix:
        """Return self + other."""
        lib = ensure_signatures()
        result = self.copy()
        status = lib.Matrix_add(
            POINTER(CMatrix)(result._c_matrix), POINTER(CMatrix)(other._c_matrix)
        )
        _check_matrix_status(status)
        return result

    def __add__(self, other: Matrix) -> Matrix:
        """Support matrix1 + matrix2."""
        return self.add(other)

    def sub(self, other: Matrix) -> Matrix:
        """Return self - other."""
        lib = ensure_signatures()
        result = self.copy()
        status = lib.Matrix_sub(
            POINTER(CMatrix)(result._c_matrix), POINTER(CMatrix)(other._c_matrix)
        )
        _check_matrix_status(status)
        return result

    def __sub__(self, other: Matrix) -> Matrix:
        """Support matrix1 - matrix2."""
        return self.sub(other)

    def scale(self, scalar: Union[float, complex]) -> Matrix:
        """Return self * scalar."""
        lib = ensure_signatures()
        result = self.copy()
        lib.Matrix_scale(
            POINTER(CMatrix)(result._c_matrix), ctypes.c_double_complex(scalar)
        )
        return result

    def __mul__(self, scalar: Union[float, complex]) -> Matrix:
        """Support matrix * scalar."""
        return self.scale(scalar)

    def __rmul__(self, scalar: Union[float, complex]) -> Matrix:
        """Support scalar * matrix."""
        return self.scale(scalar)

    def matmul(self, other: Union[Matrix, Vector]) -> Union[Matrix, Vector]:
        """Matrix multiplication: self @ other."""
        lib = ensure_signatures()

        if isinstance(other, Matrix):
            result = Matrix(self.rows, other.cols, self.data_type)
            status = lib.Matrix_Matrix_dot(
                POINTER(CMatrix)(result._c_matrix),
                POINTER(CMatrix)(self._c_matrix),
                POINTER(CMatrix)(other._c_matrix),
            )
            _check_matrix_status(status)
            return result
        elif isinstance(other, Vector):
            result = Vector(self.rows, data_type=self.data_type)
            status = lib.Matrix_Vector_dot(
                POINTER(CVector)(result._c_vector),
                POINTER(CMatrix)(self._c_matrix),
                POINTER(CVector)(other._c_vector),
            )
            _check_matrix_status(status)
            return result
        else:
            raise TypeError(f"Cannot multiply Matrix with {type(other)}")

    def __matmul__(self, other: Union[Matrix, Vector]) -> Union[Matrix, Vector]:
        """Support matrix @ other."""
        return self.matmul(other)

    def inverse(self) -> Matrix:
        """Return the inverse of this matrix."""
        lib = ensure_signatures()
        result = Matrix(self.rows, self.cols, self.data_type)
        status = lib.Matrix_inverse(
            POINTER(CMatrix)(result._c_matrix), POINTER(CMatrix)(self._c_matrix)
        )
        _check_matrix_status(status)
        return result

    def solve(self, rhs: Union[Vector, Matrix]) -> Union[Vector, Matrix]:
        """Solve this matrix times x = rhs for x."""
        lib = ensure_signatures()

        if isinstance(rhs, Vector):
            result = Vector(self.cols, data_type=self.data_type)
            status = lib.Matrix_Vector_solve(
                POINTER(CVector)(result._c_vector),
                POINTER(CMatrix)(self._c_matrix),
                POINTER(CVector)(rhs._c_vector),
            )
            _check_matrix_status(status)
            return result
        elif isinstance(rhs, Matrix):
            result = Matrix(self.cols, rhs.cols, self.data_type)
            status = lib.Matrix_Matrix_solve(
                POINTER(CMatrix)(result._c_matrix),
                POINTER(CMatrix)(self._c_matrix),
                POINTER(CMatrix)(rhs._c_matrix),
            )
            _check_matrix_status(status)
            return result
        else:
            raise TypeError(f"Cannot solve with RHS of type {type(rhs)}")

    def qr(self) -> tuple[Matrix, Matrix]:
        """Return QR decomposition as (Q, R)."""
        lib = ensure_signatures()
        Q = Matrix(self.rows, self.rows, self.data_type)
        R = Matrix(self.rows, self.cols, self.data_type)
        status = lib.Matrix_QR(
            POINTER(CMatrix)(Q._c_matrix),
            POINTER(CMatrix)(R._c_matrix),
            POINTER(CMatrix)(self._c_matrix),
        )
        _check_matrix_status(status)
        return Q, R

    def eigenvalues(self, sort: bool = False) -> Vector:
        """Return eigenvalues as a vector."""
        lib = ensure_signatures()
        evals = Vector(self.rows, data_type=self.data_type)
        status = lib.Matrix_eigvals(
            POINTER(CVector)(evals._c_vector), POINTER(CMatrix)(self._c_matrix), sort
        )
        _check_matrix_status(status)
        return evals

    def norm(self, norm_type: int = NormType.FROBENIUS) -> float:
        """Compute matrix norm."""
        lib = ensure_signatures()
        result = ctypes.c_double()
        lib.Matrix_norm(
            ctypes.byref(result), POINTER(CMatrix)(self._c_matrix), norm_type
        )
        return result.value

    def max(self) -> Union[float, complex]:
        """Return maximum element."""
        lib = ensure_signatures()
        element_size = _get_element_size(self.data_type)
        buffer = ctypes.create_string_buffer(element_size)
        lib.Matrix_max(
            ctypes.cast(buffer, ctypes.c_void_p), POINTER(CMatrix)(self._c_matrix)
        )
        return _get_scalar_value(ctypes.cast(buffer, ctypes.c_void_p), self.data_type)

    def min(self) -> Union[float, complex]:
        """Return minimum element."""
        lib = ensure_signatures()
        element_size = _get_element_size(self.data_type)
        buffer = ctypes.create_string_buffer(element_size)
        lib.Matrix_min(
            ctypes.cast(buffer, ctypes.c_void_p), POINTER(CMatrix)(self._c_matrix)
        )
        return _get_scalar_value(ctypes.cast(buffer, ctypes.c_void_p), self.data_type)

    def print(self) -> None:
        """Print matrix to stdout."""
        lib = ensure_signatures()
        lib.Matrix_print(POINTER(CMatrix)(self._c_matrix))

    def __repr__(self) -> str:
        return f"Matrix(shape={self.shape}, dtype={self.data_type})"


class Vector:
    """
    High-level Vector wrapper.

    Wraps the low-level CVector struct and provides Pythonic operations
    for vector manipulation and linear algebra.
    """

    def __init__(
        self,
        dim: int,
        data_type: int = DataType.REAL,
        init_values: Optional[Iterable] = None,
    ):
        """
        Initialize a Vector.

        Args:
            dim: Dimension of the vector
            data_type: DataType enum (INT, REAL, or COMPLEX)
            init_values: Optional iterable of initial values
        """
        self.data_type = data_type
        self._c_vector = CVector()

        lib = ensure_signatures()

        if init_values is not None:
            values = list(init_values)
            c_array = _python_to_c_array(values, data_type)
            self._c_vector = lib.Vector_new_vals(dim, c_array, data_type)
        else:
            self._c_vector = lib.Vector_new(dim, None, data_type)

    def __del__(self):
        """Cleanup: free the underlying C vector."""
        try:
            lib = ensure_signatures()
            lib.Vector_free(POINTER(CVector)(self._c_vector))
        except:
            pass

    @property
    def dim(self) -> int:
        """Return dimension of the vector."""
        return self._c_vector.dim

    @classmethod
    def zeros(cls, dim: int, data_type: int = DataType.REAL) -> Vector:
        """Create a zero vector of given dimension."""
        lib = ensure_signatures()
        vec = cls.__new__(cls)
        vec.data_type = data_type
        vec._c_vector = lib.Vector_zeros(dim, data_type)
        return vec

    @classmethod
    def ones(cls, dim: int, data_type: int = DataType.REAL) -> Vector:
        """Create a vector of all ones."""
        lib = ensure_signatures()
        vec = cls.__new__(cls)
        vec.data_type = data_type
        vec._c_vector = lib.Vector_ones(dim, data_type)
        return vec

    @classmethod
    def random_normal(
        cls,
        dim: int,
        mean: float = 0.0,
        variance: float = 1.0,
        data_type: int = DataType.REAL,
    ) -> Vector:
        """Create a vector with random values from normal distribution."""
        lib = ensure_signatures()
        vec = cls.__new__(cls)
        vec.data_type = data_type
        vec._c_vector = lib.Vector_new_random_normal(dim, mean, variance, data_type)
        return vec

    @classmethod
    def random_uniform(
        cls,
        dim: int,
        min_val: float = 0.0,
        max_val: float = 1.0,
        data_type: int = DataType.REAL,
    ) -> Vector:
        """Create a vector with random values from uniform distribution."""
        lib = ensure_signatures()
        vec = cls.__new__(cls)
        vec.data_type = data_type
        vec._c_vector = lib.Vector_new_random_uniform(dim, min_val, max_val, data_type)
        return vec

    def copy(self) -> Vector:
        """Return a deep copy of this vector."""
        lib = ensure_signatures()
        result = Vector(self.dim, self.data_type)
        lib.Vector_copy(
            POINTER(CVector)(result._c_vector), POINTER(CVector)(self._c_vector)
        )
        return result

    def __getitem__(self, index: int) -> Any:
        """Get element at index."""
        lib = ensure_signatures()
        element_size = _get_element_size(self.data_type)
        buffer = ctypes.create_string_buffer(element_size)
        status = lib.Vector_at(
            ctypes.cast(buffer, ctypes.c_void_p),
            POINTER(CVector)(self._c_vector),
            index,
        )
        _check_vector_status(status)
        return _get_scalar_value(ctypes.cast(buffer, ctypes.c_void_p), self.data_type)

    def __setitem__(self, index: int, value: Union[int, float, complex]) -> None:
        """Set element at index."""
        lib = ensure_signatures()
        c_value = _python_to_c_array([value], self.data_type)
        status = lib.Vector_set_item(POINTER(CVector)(self._c_vector), index, c_value)
        _check_vector_status(status)

    def add(self, other: Vector) -> Vector:
        """Return self + other."""
        lib = ensure_signatures()
        result = self.copy()
        status = lib.Vector_add(
            POINTER(CVector)(result._c_vector), POINTER(CVector)(other._c_vector)
        )
        _check_vector_status(status)
        return result

    def __add__(self, other: Vector) -> Vector:
        """Support vector1 + vector2."""
        return self.add(other)

    def sub(self, other: Vector) -> Vector:
        """Return self - other."""
        lib = ensure_signatures()
        result = self.copy()
        status = lib.Vector_sub(
            POINTER(CVector)(result._c_vector), POINTER(CVector)(other._c_vector)
        )
        _check_vector_status(status)
        return result

    def __sub__(self, other: Vector) -> Vector:
        """Support vector1 - vector2."""
        return self.sub(other)

    def scale(self, scalar: Union[int, float, complex]) -> Vector:
        """Return self * scalar."""
        lib = ensure_signatures()
        result = self.copy()
        c_scalar = _python_to_c_array([scalar], self.data_type)
        status = lib.Vector_scale(POINTER(CVector)(result._c_vector), c_scalar)
        _check_vector_status(status)
        return result

    def __mul__(self, scalar: Union[int, float, complex]) -> Vector:
        """Support vector * scalar."""
        return self.scale(scalar)

    def __rmul__(self, scalar: Union[int, float, complex]) -> Vector:
        """Support scalar * vector."""
        return self.scale(scalar)

    def dot(self, other: Vector) -> Union[float, complex]:
        """Compute dot product with another vector."""
        lib = ensure_signatures()
        element_size = _get_element_size(self.data_type)
        buffer = ctypes.create_string_buffer(element_size)
        status = lib.Vector_dot(
            ctypes.cast(buffer, ctypes.c_void_p),
            POINTER(CVector)(self._c_vector),
            POINTER(CVector)(other._c_vector),
        )
        _check_vector_status(status)
        return _get_scalar_value(ctypes.cast(buffer, ctypes.c_void_p), self.data_type)

    def norm(self) -> float:
        """Compute L2 norm of the vector."""
        lib = ensure_signatures()
        return lib.Vector_norm(POINTER(CVector)(self._c_vector))

    def max(self) -> Union[int, float, complex]:
        """Return maximum element."""
        lib = ensure_signatures()
        element_size = _get_element_size(self.data_type)
        buffer = ctypes.create_string_buffer(element_size)
        lib.Vector_max(
            ctypes.cast(buffer, ctypes.c_void_p), POINTER(CVector)(self._c_vector)
        )
        return _get_scalar_value(ctypes.cast(buffer, ctypes.c_void_p), self.data_type)

    def min(self) -> Union[int, float, complex]:
        """Return minimum element."""
        lib = ensure_signatures()
        element_size = _get_element_size(self.data_type)
        buffer = ctypes.create_string_buffer(element_size)
        lib.Vector_min(
            ctypes.cast(buffer, ctypes.c_void_p), POINTER(CVector)(self._c_vector)
        )
        return _get_scalar_value(ctypes.cast(buffer, ctypes.c_void_p), self.data_type)

    def sort(self) -> Vector:
        """Return a sorted copy of this vector."""
        lib = ensure_signatures()
        result = Vector(self.dim, self.data_type)
        status = lib.Vector_sort(
            POINTER(CVector)(result._c_vector), POINTER(CVector)(self._c_vector)
        )
        _check_vector_status(status)
        return result

    def sort_inplace(self) -> None:
        """Sort this vector in place."""
        lib = ensure_signatures()
        status = lib.Vector_sort_inplace(POINTER(CVector)(self._c_vector))
        _check_vector_status(status)

    def print(self) -> None:
        """Print vector to stdout."""
        lib = ensure_signatures()
        lib.Vector_print(POINTER(CVector)(self._c_vector))

    def __repr__(self) -> str:
        return f"Vector(dim={self.dim}, dtype={self.data_type})"
