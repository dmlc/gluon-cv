# coding: utf-8
"""Unified backend configuration module for GluonCV.

This module provides a centralized way to manage and switch between
MXNet and PyTorch backends at runtime.

Example usage:
    >>> import gluoncv as gcv
    >>> gcv.set_backend('pytorch')
    >>> gcv.get_backend()
    'pytorch'
    >>> with gcv.backend_context('mxnet'):
    ...     pass
"""
from __future__ import absolute_import

import threading
import warnings
from contextlib import contextmanager
from enum import Enum


class Backend(Enum):
    MXNET = 'mxnet'
    PYTORCH = 'pytorch'
    AUTO = 'auto'


class BackendState:
    MXNET_AVAILABLE = 'mxnet_available'
    PYTORCH_AVAILABLE = 'pytorch_available'
    MXNET_VERSION = 'mxnet_version'
    PYTORCH_VERSION = 'pytorch_version'


class BackendManager:
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if BackendManager._initialized:
            return
        with BackendManager._lock:
            if BackendManager._initialized:
                return
            
            self._backend = None
            self._locked = False
            self._auto_detected = False
            
            self._mxnet_available = False
            self._pytorch_available = False
            self._mxnet_version = None
            self._pytorch_version = None
            self._mxnet_module = None
            self._pytorch_module = None
            
            self._detect_available_backends()
            BackendManager._initialized = True

    def _detect_available_backends(self):
        self._mxnet_available = False
        self._pytorch_available = False
        
        try:
            import mxnet as mx
            from distutils.version import LooseVersion
            mx_version = mx.__version__
            if LooseVersion(mx_version) >= LooseVersion('1.4.0') and \
               LooseVersion(mx_version) < LooseVersion('2.0.0'):
                self._mxnet_available = True
                self._mxnet_version = mx_version
                self._mxnet_module = mx
        except ImportError:
            pass
        
        try:
            import torch
            from distutils.version import LooseVersion
            torch_version = torch.__version__
            if LooseVersion(torch_version) >= LooseVersion('1.4.0') and \
               LooseVersion(torch_version) < LooseVersion('2.0.0'):
                self._pytorch_available = True
                self._pytorch_version = torch_version
                self._pytorch_module = torch
        except ImportError:
            pass
        
        self._auto_detected = True

    def _auto_select_backend(self):
        if self._mxnet_available and self._pytorch_available:
            return Backend.MXNET
        elif self._mxnet_available:
            return Backend.MXNET
        elif self._pytorch_available:
            return Backend.PYTORCH
        else:
            raise ImportError(
                'Unable to import modules due to missing `mxnet` & `torch`. '
                'You should install at least one deep learning framework.'
            )

    @property
    def backend(self):
        if self._backend is None:
            self._backend = self._auto_select_backend()
        return self._backend

    @property
    def is_locked(self):
        return self._locked

    def get_backend(self):
        return self.backend.value

    def set_backend(self, backend, lock=False):
        if self._locked:
            raise RuntimeError(
                f"Backend is locked to '{self._backend.value}'. "
                "Unlock it first using unlock_backend() before switching."
            )
        
        if isinstance(backend, str):
            backend = Backend(backend.lower())
        elif isinstance(backend, Backend):
            pass
        else:
            raise ValueError(
                f"Invalid backend: {backend}. "
                f"Must be one of: {[b.value for b in Backend]}"
            )
        
        if backend == Backend.AUTO:
            backend = self._auto_select_backend()
        
        if backend == Backend.MXNET and not self._mxnet_available:
            raise ImportError(
                "MXNet backend is not available. "
                "Please install MXNet >=1.4.0,<2.0.0 first."
            )
        
        if backend == Backend.PYTORCH and not self._pytorch_available:
            raise ImportError(
                "PyTorch backend is not available. "
                "Please install PyTorch >=1.4.0,<2.0.0 first."
            )
        
        old_backend = self._backend
        self._backend = backend
        
        if lock:
            self._locked = True
        
        if old_backend is not None and old_backend != backend:
            if self._mxnet_available and self._pytorch_available:
                warnings.warn(
                    f"Backend switched from '{old_backend.value}' to '{backend.value}'. "
                    "Note that both frameworks are installed, which may increase GPU memory footprint."
                )
        
        return self._backend

    def lock_backend(self):
        if self._backend is None:
            self._backend = self._auto_select_backend()
        self._locked = True

    def unlock_backend(self):
        self._locked = False

    def is_available(self, backend):
        if isinstance(backend, str):
            backend = Backend(backend.lower())
        
        if backend == Backend.MXNET:
            return self._mxnet_available
        elif backend == Backend.PYTORCH:
            return self._pytorch_available
        elif backend == Backend.AUTO:
            return self._mxnet_available or self._pytorch_available
        return False

    def get_version(self, backend):
        if isinstance(backend, str):
            backend = Backend(backend.lower())
        
        if backend == Backend.MXNET:
            return self._mxnet_version
        elif backend == Backend.PYTORCH:
            return self._pytorch_version
        return None

    def get_module(self, backend=None):
        if backend is None:
            backend = self.backend
        elif isinstance(backend, str):
            backend = Backend(backend.lower())
        
        if backend == Backend.MXNET:
            if self._mxnet_module is None:
                raise ImportError("MXNet is not available.")
            return self._mxnet_module
        elif backend == Backend.PYTORCH:
            if self._pytorch_module is None:
                raise ImportError("PyTorch is not available.")
            return self._pytorch_module
        return None

    @property
    def mxnet_available(self):
        return self._mxnet_available

    @property
    def pytorch_available(self):
        return self._pytorch_available

    @property
    def mxnet_version(self):
        return self._mxnet_version

    @property
    def pytorch_version(self):
        return self._pytorch_version

    def __repr__(self):
        locked_str = " (locked)" if self._locked else ""
        return (
            f"BackendManager(backend={self.backend.value}{locked_str}, "
            f"mxnet={self._mxnet_available}, pytorch={self._pytorch_available})"
        )


_manager = BackendManager()


def get_backend():
    """Get the current backend name.
    
    Returns
    -------
    str
        Current backend name ('mxnet' or 'pytorch')
    """
    return _manager.get_backend()


def set_backend(backend, lock=False):
    """Set the global backend.
    
    Parameters
    ----------
    backend : str or Backend
        Backend to use ('mxnet', 'pytorch', or 'auto')
    lock : bool, default False
        If True, lock the backend to prevent further changes
    
    Returns
    -------
    Backend
        The selected backend
    
    Raises
    ------
    ImportError
        If the requested backend is not available
    RuntimeError
        If the backend is locked and a different backend is requested
    """
    return _manager.set_backend(backend, lock=lock)


def lock_backend():
    """Lock the current backend to prevent further changes."""
    _manager.lock_backend()


def unlock_backend():
    """Unlock the backend to allow changes."""
    _manager.unlock_backend()


def is_backend_locked():
    """Check if the backend is locked.
    
    Returns
    -------
    bool
        True if the backend is locked
    """
    return _manager.is_locked


def is_available(backend):
    """Check if a backend is available.
    
    Parameters
    ----------
    backend : str or Backend
        Backend to check ('mxnet', 'pytorch', or 'auto')
    
    Returns
    -------
    bool
        True if the backend is available
    """
    return _manager.is_available(backend)


def get_version(backend):
    """Get the version of a backend.
    
    Parameters
    ----------
    backend : str or Backend
        Backend to check ('mxnet' or 'pytorch')
    
    Returns
    -------
    str or None
        Version string or None if not available
    """
    return _manager.get_version(backend)


def get_module(backend=None):
    """Get the module for a backend.
    
    Parameters
    ----------
    backend : str, Backend, or None
        Backend to get module for. If None, uses current backend.
    
    Returns
    -------
    module
        The mxnet or torch module
    
    Raises
    ------
    ImportError
        If the backend is not available
    """
    return _manager.get_module(backend)


@contextmanager
def backend_context(backend, lock=False):
    """Context manager for temporarily switching backend.
    
    Parameters
    ----------
    backend : str or Backend
        Backend to use within the context
    lock : bool, default False
        If True, lock the backend within the context
    
    Example
    -------
    >>> with gluoncv.backend_context('pytorch'):
    ...     pass
    """
    old_locked = _manager.is_locked
    old_backend = _manager.backend
    
    if old_locked:
        _manager.unlock_backend()
    
    try:
        _manager.set_backend(backend, lock=lock)
        yield
    finally:
        if not lock:
            if old_locked:
                _manager._backend = old_backend
                _manager.lock_backend()
            else:
                _manager._backend = old_backend


def require_backend(backend):
    """Require a specific backend, raising an error if different.
    
    Parameters
    ----------
    backend : str or Backend
        Required backend
    
    Raises
    ------
    RuntimeError
        If current backend is different from required
    """
    if isinstance(backend, str):
        backend = Backend(backend.lower())
    
    current = _manager.backend
    if current != backend:
        raise RuntimeError(
            f"This operation requires '{backend.value}' backend, "
            f"but current backend is '{current.value}'. "
            f"Use gluoncv.set_backend('{backend.value}') to switch."
        )


def check_backend_available(backend):
    """Check if a backend is available, raising an error if not.
    
    Parameters
    ----------
    backend : str or Backend
        Backend to check
    
    Raises
    ------
    ImportError
        If the backend is not available
    """
    if isinstance(backend, str):
        backend = Backend(backend.lower())
    
    if not _manager.is_available(backend):
        raise ImportError(
            f"'{backend.value}' backend is not available. "
            f"Please install it first."
        )
