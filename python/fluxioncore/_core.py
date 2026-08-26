"""
FluxionCore ctypes bindings to the C numerical library.

This module provides low-level ctypes bindings to matrix, vector, and Fourier
transformation functions implemented in C. It handles:
- Shared library loading
- Type definitions for C structs (Matrix, Vector, DataPoints)
- Function signatures and error codes
- Memory management callbacks
"""

import ctypes
import os
from ctypes import POINTER, c_size_t, c_double, c_int, c_void_p, c_bool
from typing import Optional, Callable
from enum import IntEnum


# ============================================================================
# Error Codes
# ============================================================================

class MatrixStatus(IntEnum):
    """Status codes returned by matrix operations."""
    SUCCESS = 0
    DIMENSION_ERROR = 1
    MATH_ERROR = 2
    BASIC_ERROR = 3


class VectorStatus(IntEnum):
    """Status codes returned by vector operations."""
    SUCCESS = 0
    DIMENSION_ERROR = 1


# ============================================================================
# Data Types
# ============================================================================

class DataType(IntEnum):
    """Supported numerical data types in the C library."""
    # Assuming from config.h that these are defined; adjust if needed
    FLOAT = 0
    DOUBLE = 1
    COMPLEX_DOUBLE = 2


class NormType(IntEnum):
    """Matrix norm types."""
    FROBENIUS = 0
    SPECTRAL = 1


class FourierAlgorithm(IntEnum):
    """Fourier transform algorithms."""
    DFT_SCALAR = 0
    DFT_PARALLEL = 1
    FFT = 2


# ============================================================================
# C Structure Definitions
# ============================================================================

class CMatrix(ctypes.Structure):
    """C-side matrix struct: void* values, size_t m, n, DataType dt."""
    _fields_ = [
        ("values", c_void_p),
        ("m", c_size_t),
        ("n", c_size_t),
        ("dt", c_int),  # DataType enum
    ]


class CVector(ctypes.Structure):
    """C-side vector struct: void* values, size_t dim, DataType dt."""
    _fields_ = [
        ("values", c_void_p),
        ("dim", c_size_t),
        ("dt", c_int),  # DataType enum
    ]


class CDataPoints(ctypes.Structure):
    """C-side Fourier data struct: double complex* data, size_t size."""
    _fields_ = [
        ("data", c_void_p),  # Pointer to double complex array
        ("size", c_size_t),
    ]


# ============================================================================
# Library Loading
# ============================================================================

def _find_library() -> str:
    """
    Locate the compiled C library (.so/.dll/.dylib).
    
    Searches in standard locations relative to this module:
    1. ../../../bin/main (if built via makefile)
    2. ../../../libmatrixvector.so (if compiled as shared library)
    3. System library path
    
    Returns:
        Path to the shared library.
    
    Raises:
        OSError: If the library cannot be found.
    """
    # Get the directory where this module is located
    module_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(module_dir))
    
    # Possible library locations
    search_paths = [
        os.path.join(repo_root, "bin", "libmatrixvector.so"),
        os.path.join(repo_root, "bin", "libmatrixvector.dylib"),
        os.path.join(repo_root, "bin", "matrixvector.dll"),
        os.path.join(repo_root, "libmatrixvector.so"),
        "libmatrixvector.so",  # System path
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            return path
    
    raise OSError(
        f"Could not find C library. Searched: {search_paths}\n"
        "Have you built the C library? Run 'make' in the repo root."
    )


def _load_library() -> ctypes.CDLL:
    """Load and return the C library."""
    lib_path = _find_library()
    lib = ctypes.CDLL(lib_path)
    return lib


# Lazy-load the library
_lib: Optional[ctypes.CDLL] = None


def get_lib() -> ctypes.CDLL:
    """Get the loaded C library, loading it if necessary."""
    global _lib
    if _lib is None:
        _lib = _load_library()
    return _lib


# ============================================================================
# Function Signatures
# ============================================================================

def _setup_function_signatures(lib: ctypes.CDLL) -> None:
    """Define ctypes signatures for all C functions."""
    
    # ========== Matrix Functions ==========
    
    # Matrix_new(size_t m, size_t n, DataType dt) -> Matrix
    lib.Matrix_new.argtypes = [c_size_t, c_size_t, c_int]
    lib.Matrix_new.restype = CMatrix
    
    # Matrix_zeros(size_t m, size_t n, DataType dt) -> Matrix
    lib.Matrix_zeros.argtypes = [c_size_t, c_size_t, c_int]
    lib.Matrix_zeros.restype = CMatrix
    
    # Matrix_new_vals(size_t m, size_t n, const void* init_vals, DataType dt) -> Matrix
    lib.Matrix_new_vals.argtypes = [c_size_t, c_size_t, c_void_p, c_int]
    lib.Matrix_new_vals.restype = CMatrix
    
    # Matrix_new_random_normal(size_t m, size_t n, double mean, double variance, DataType dt) -> Matrix
    lib.Matrix_new_random_normal.argtypes = [c_size_t, c_size_t, c_double, c_double, c_int]
    lib.Matrix_new_random_normal.restype = CMatrix
    
    # Matrix_new_random_uniform(size_t m, size_t n, double min, double max, DataType dt) -> Matrix
    lib.Matrix_new_random_uniform.argtypes = [c_size_t, c_size_t, c_double, c_double, c_int]
    lib.Matrix_new_random_uniform.restype = CMatrix
    
    # Matrix_new_random_symmetric(size_t n, DataType dt) -> Matrix
    lib.Matrix_new_random_symmetric.argtypes = [c_size_t, c_int]
    lib.Matrix_new_random_symmetric.restype = CMatrix
    
    # Matrix_diag(const Vector* v) -> Matrix
    lib.Matrix_diag.argtypes = [POINTER(CVector)]
    lib.Matrix_diag.restype = CMatrix
    
    # Matrix_diag_val(size_t n, const void* val, DataType dt) -> Matrix
    lib.Matrix_diag_val.argtypes = [c_size_t, c_void_p, c_int]
    lib.Matrix_diag_val.restype = CMatrix
    
    # Matrix_copy(Matrix* target, const Matrix* M) -> void
    lib.Matrix_copy.argtypes = [POINTER(CMatrix), POINTER(CMatrix)]
    lib.Matrix_copy.restype = None
    
    # Matrix_free(Matrix* M) -> void
    lib.Matrix_free.argtypes = [POINTER(CMatrix)]
    lib.Matrix_free.restype = None
    
    # Matrix_print(const Matrix* M) -> void
    lib.Matrix_print.argtypes = [POINTER(CMatrix)]
    lib.Matrix_print.restype = None
    
    # Matrix_set_at(Matrix* M, size_t m, size_t n, const void* value) -> int
    lib.Matrix_set_at.argtypes = [POINTER(CMatrix), c_size_t, c_size_t, c_void_p]
    lib.Matrix_set_at.restype = c_int
    
    # Matrix_get_at(void* target, const Matrix* M, size_t m, size_t n) -> int
    lib.Matrix_get_at.argtypes = [c_void_p, POINTER(CMatrix), c_size_t, c_size_t]
    lib.Matrix_get_at.restype = c_int
    
    # Matrix_add(Matrix* target, const Matrix* M) -> int
    lib.Matrix_add.argtypes = [POINTER(CMatrix), POINTER(CMatrix)]
    lib.Matrix_add.restype = c_int
    
    # Matrix_sub(Matrix* target, const Matrix* M) -> int
    lib.Matrix_sub.argtypes = [POINTER(CMatrix), POINTER(CMatrix)]
    lib.Matrix_sub.restype = c_int
    
    # Matrix_scale(Matrix* target, complex double lambda) -> void
    lib.Matrix_scale.argtypes = [POINTER(CMatrix), ctypes.c_complex]
    lib.Matrix_scale.restype = None
    
    # Matrix_Matrix_dot(Matrix* target, const Matrix* M1, const Matrix* M2) -> int
    lib.Matrix_Matrix_dot.argtypes = [POINTER(CMatrix), POINTER(CMatrix), POINTER(CMatrix)]
    lib.Matrix_Matrix_dot.restype = c_int
    
    # Matrix_Vector_dot(Vector* target, const Matrix* M, const Vector* v) -> int
    lib.Matrix_Vector_dot.argtypes = [POINTER(CVector), POINTER(CMatrix), POINTER(CVector)]
    lib.Matrix_Vector_dot.restype = c_int
    
    # Matrix_inverse(Matrix* target, const Matrix* M) -> int
    lib.Matrix_inverse.argtypes = [POINTER(CMatrix), POINTER(CMatrix)]
    lib.Matrix_inverse.restype = c_int
    
    # Matrix_Vector_solve(Vector* target, const Matrix* M, const Vector* v) -> int
    lib.Matrix_Vector_solve.argtypes = [POINTER(CVector), POINTER(CMatrix), POINTER(CVector)]
    lib.Matrix_Vector_solve.restype = c_int
    
    # Matrix_Matrix_solve(Matrix* target, const Matrix* M, const Matrix* N) -> int
    lib.Matrix_Matrix_solve.argtypes = [POINTER(CMatrix), POINTER(CMatrix), POINTER(CMatrix)]
    lib.Matrix_Matrix_solve.restype = c_int
    
    # Matrix_QR(Matrix* Q, Matrix* R, const Matrix* A) -> int
    lib.Matrix_QR.argtypes = [POINTER(CMatrix), POINTER(CMatrix), POINTER(CMatrix)]
    lib.Matrix_QR.restype = c_int
    
    # Matrix_eigvals(Vector* eigenvalues, const Matrix* M, bool sort) -> int
    lib.Matrix_eigvals.argtypes = [POINTER(CVector), POINTER(CMatrix), c_bool]
    lib.Matrix_eigvals.restype = c_int
    
    # Matrix_norm(void* target, const Matrix* M, NormType nt) -> void
    lib.Matrix_norm.argtypes = [c_void_p, POINTER(CMatrix), c_int]
    lib.Matrix_norm.restype = None
    
    # Matrix_max(void* target, const Matrix* M) -> void
    lib.Matrix_max.argtypes = [c_void_p, POINTER(CMatrix)]
    lib.Matrix_max.restype = None
    
    # Matrix_min(void* target, const Matrix* M) -> void
    lib.Matrix_min.argtypes = [c_void_p, POINTER(CMatrix)]
    lib.Matrix_min.restype = None
    
    # Matrix_Hessenberg(Matrix* H, const Matrix* A) -> int
    lib.Matrix_Hessenberg.argtypes = [POINTER(CMatrix), POINTER(CMatrix)]
    lib.Matrix_Hessenberg.restype = c_int
    
    # Matrix_Vector_outer(Matrix* target, const Vector* a, const Vector* b) -> int
    lib.Matrix_Vector_outer.argtypes = [POINTER(CMatrix), POINTER(CVector), POINTER(CVector)]
    lib.Matrix_Vector_outer.restype = c_int
    
    # Matrix_Hadamard_dot(Matrix* target, const Matrix* A, const Matrix* B) -> int
    lib.Matrix_Hadamard_dot.argtypes = [POINTER(CMatrix), POINTER(CMatrix), POINTER(CMatrix)]
    lib.Matrix_Hadamard_dot.restype = c_int
    
    # ========== Vector Functions ==========
    
    # Vector_new(size_t dim, const void* init_val, DataType dt) -> Vector
    lib.Vector_new.argtypes = [c_size_t, c_void_p, c_int]
    lib.Vector_new.restype = CVector
    
    # Vector_zeros(size_t dim, DataType dt) -> Vector
    lib.Vector_zeros.argtypes = [c_size_t, c_int]
    lib.Vector_zeros.restype = CVector
    
    # Vector_ones(size_t dim, DataType dt) -> Vector
    lib.Vector_ones.argtypes = [c_size_t, c_int]
    lib.Vector_ones.restype = CVector
    
    # Vector_new_vals(size_t dim, const void* init_vals, DataType dt) -> Vector
    lib.Vector_new_vals.argtypes = [c_size_t, c_void_p, c_int]
    lib.Vector_new_vals.restype = CVector
    
    # Vector_new_random_normal(size_t dim, double mean, double variance, DataType dt) -> Vector
    lib.Vector_new_random_normal.argtypes = [c_size_t, c_double, c_double, c_int]
    lib.Vector_new_random_normal.restype = CVector
    
    # Vector_new_random_uniform(size_t dim, double min, double max, DataType dt) -> Vector
    lib.Vector_new_random_uniform.argtypes = [c_size_t, c_double, c_double, c_int]
    lib.Vector_new_random_uniform.restype = CVector
    
    # Vector_new_copy(const Vector* v) -> Vector
    lib.Vector_new_copy.argtypes = [POINTER(CVector)]
    lib.Vector_new_copy.restype = CVector
    
    # Vector_zeros_like(const Vector* v) -> Vector
    lib.Vector_zeros_like.argtypes = [POINTER(CVector)]
    lib.Vector_zeros_like.restype = CVector
    
    # Vector_copy(Vector* target, const Vector* v) -> void
    lib.Vector_copy.argtypes = [POINTER(CVector), POINTER(CVector)]
    lib.Vector_copy.restype = None
    
    # Vector_free(Vector* v) -> void
    lib.Vector_free.argtypes = [POINTER(CVector)]
    lib.Vector_free.restype = None
    
    # Vector_print(Vector* v) -> void
    lib.Vector_print.argtypes = [POINTER(CVector)]
    lib.Vector_print.restype = None
    
    # Vector_at(void* target, const Vector* v, size_t index) -> int
    lib.Vector_at.argtypes = [c_void_p, POINTER(CVector), c_size_t]
    lib.Vector_at.restype = c_int
    
    # Vector_set_item(Vector* v, size_t index, const void* val) -> int
    lib.Vector_set_item.argtypes = [POINTER(CVector), c_size_t, c_void_p]
    lib.Vector_set_item.restype = c_int
    
    # Vector_add(Vector* target, const Vector* v) -> int
    lib.Vector_add.argtypes = [POINTER(CVector), POINTER(CVector)]
    lib.Vector_add.restype = c_int
    
    # Vector_sub(Vector* target, const Vector* v) -> int
    lib.Vector_sub.argtypes = [POINTER(CVector), POINTER(CVector)]
    lib.Vector_sub.restype = c_int
    
    # Vector_scale(Vector* target, const void* lambda) -> int
    lib.Vector_scale.argtypes = [POINTER(CVector), c_void_p]
    lib.Vector_scale.restype = c_int
    
    # Vector_dot(void* target, const Vector* u, const Vector* v) -> int
    lib.Vector_dot.argtypes = [c_void_p, POINTER(CVector), POINTER(CVector)]
    lib.Vector_dot.restype = c_int
    
    # Vector_norm(const Vector* v) -> real_t (assume double)
    lib.Vector_norm.argtypes = [POINTER(CVector)]
    lib.Vector_norm.restype = c_double
    
    # Vector_max(void* target, const Vector* v) -> void
    lib.Vector_max.argtypes = [c_void_p, POINTER(CVector)]
    lib.Vector_max.restype = None
    
    # Vector_min(void* target, const Vector* v) -> void
    lib.Vector_min.argtypes = [c_void_p, POINTER(CVector)]
    lib.Vector_min.restype = None
    
    # Vector_sort(Vector* target, const Vector* v) -> int
    lib.Vector_sort.argtypes = [POINTER(CVector), POINTER(CVector)]
    lib.Vector_sort.restype = c_int
    
    # Vector_sort_inplace(Vector* v) -> int
    lib.Vector_sort_inplace.argtypes = [POINTER(CVector)]
    lib.Vector_sort_inplace.restype = c_int
    
    # ========== Fourier Functions ==========
    
    # fourier_transform(DataPoints* target, const DataPoints* source, bool to) -> int
    lib.fourier_transform.argtypes = [POINTER(CDataPoints), POINTER(CDataPoints), c_bool]
    lib.fourier_transform.restype = c_int
    
    # DataPoints_new(size_t size) -> DataPoints
    lib.DataPoints_new.argtypes = [c_size_t]
    lib.DataPoints_new.restype = CDataPoints
    
    # DataPoints_free(DataPoints* data) -> int
    lib.DataPoints_free.argtypes = [POINTER(CDataPoints)]
    lib.DataPoints_free.restype = c_int
    
    # ========== Config & Init ==========
    
    # config_init() -> void
    lib.config_init.argtypes = []
    lib.config_init.restype = None
    
    # Matrix_init_prng(int seed) -> void
    lib.Matrix_init_prng.argtypes = [c_int]
    lib.Matrix_init_prng.restype = None
    
    # Vector_init_prng(int seed) -> void
    lib.Vector_init_prng.argtypes = [c_int]
    lib.Vector_init_prng.restype = None


# Setup on first library load
_signatures_loaded = False


def ensure_signatures() -> ctypes.CDLL:
    """Ensure function signatures are set up before use."""
    global _signatures_loaded
    lib = get_lib()
    if not _signatures_loaded:
        _setup_function_signatures(lib)
        _signatures_loaded = True
    return lib


# ============================================================================
# Helper Functions for Type Conversion
# ============================================================================

def _get_element_size(dt: int) -> int:
    """Get the size in bytes of a single element for a DataType."""
    if dt == DataType.FLOAT:
        return 4  # sizeof(float)
    elif dt == DataType.DOUBLE:
        return 8  # sizeof(double)
    elif dt == DataType.COMPLEX_DOUBLE:
        return 16  # sizeof(double complex) = 2 * sizeof(double)
    else:
        raise ValueError(f"Unknown DataType: {dt}")


def _python_to_c_array(data: list, dt: int) -> c_void_p:
    """
    Convert Python list to ctypes array and return as void pointer.
    
    Args:
        data: Python list of values
        dt: DataType enum value
    
    Returns:
        ctypes void pointer to allocated array
    """
    element_size = _get_element_size(dt)
    
    if dt == DataType.FLOAT:
        c_array = (ctypes.c_float * len(data))(*data)
    elif dt == DataType.DOUBLE:
        c_array = (ctypes.c_double * len(data))(*data)
    elif dt == DataType.COMPLEX_DOUBLE:
        c_array = (ctypes.c_complex * len(data))(*data)
    else:
        raise ValueError(f"Unknown DataType: {dt}")
    
    return ctypes.cast(c_array, c_void_p)


def _c_array_to_python(ptr: c_void_p, size: int, dt: int) -> list:
    """
    Convert C array back to Python list.
    
    Args:
        ptr: Void pointer to C array
        size: Number of elements
        dt: DataType enum value
    
    Returns:
        Python list of values
    """
    if ptr is None or ptr == 0:
        return []
    
    if dt == DataType.FLOAT:
        c_array = ctypes.cast(ptr, POINTER(ctypes.c_float * size))
    elif dt == DataType.DOUBLE:
        c_array = ctypes.cast(ptr, POINTER(ctypes.c_double * size))
    elif dt == DataType.COMPLEX_DOUBLE:
        c_array = ctypes.cast(ptr, POINTER(ctypes.c_complex * size))
    else:
        raise ValueError(f"Unknown DataType: {dt}")
    
    return list(c_array.contents)


def _get_scalar_value(ptr: c_void_p, dt: int):
    """Extract a single scalar value from a void pointer."""
    if ptr is None or ptr == 0:
        return None
    
    if dt == DataType.FLOAT:
        return ctypes.cast(ptr, POINTER(ctypes.c_float))[0]
    elif dt == DataType.DOUBLE:
        return ctypes.cast(ptr, POINTER(ctypes.c_double))[0]
    elif dt == DataType.COMPLEX_DOUBLE:
        return ctypes.cast(ptr, POINTER(ctypes.c_complex))[0]
    else:
        raise ValueError(f"Unknown DataType: {dt}")


# ============================================================================
# Exception Classes
# ============================================================================

class FluxionError(Exception):
    """Base exception for FluxionCore errors."""
    pass


class DimensionError(FluxionError):
    """Raised when matrix/vector dimensions don't match for an operation."""
    pass


class MathError(FluxionError):
    """Raised for numerical/mathematical errors (singular matrix, etc.)."""
    pass


class BasicError(FluxionError):
    """Raised for basic operation errors."""
    pass


def _check_matrix_status(status: int) -> None:
    """Check a MatrixStatus code and raise exception if needed."""
    if status == MatrixStatus.SUCCESS:
        return
    elif status == MatrixStatus.DIMENSION_ERROR:
        raise DimensionError("Matrix dimension mismatch")
    elif status == MatrixStatus.MATH_ERROR:
        raise MathError("Mathematical error in matrix operation")
    elif status == MatrixStatus.BASIC_ERROR:
        raise BasicError("Basic error in matrix operation")
    else:
        raise FluxionError(f"Unknown matrix status code: {status}")


def _check_vector_status(status: int) -> None:
    """Check a VectorStatus code and raise exception if needed."""
    if status == VectorStatus.SUCCESS:
        return
    elif status == VectorStatus.DIMENSION_ERROR:
        raise DimensionError("Vector dimension mismatch")
    else:
        raise FluxionError(f"Unknown vector status code: {status}")
