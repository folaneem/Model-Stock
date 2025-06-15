"""
Error handling utilities for the application.

This module provides decorators and context managers for consistent
error handling and fallback behavior throughout the application.
"""

import functools
import logging
import time
from typing import Callable, TypeVar, Any, Optional, Type, Dict, List, Union
import streamlit as st

logger = logging.getLogger(__name__)

T = TypeVar('T')

class ErrorHandlingConfig:
    """Configuration for error handling behavior."""
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 30.0,
        backoff_factor: float = 2.0,
        silent: bool = False,
        default_return: Any = None,
        allowed_exceptions: Optional[tuple] = None,
        log_errors: bool = True,
        show_user_message: bool = True,
        user_message_prefix: str = ""
    ):
        """
        Initialize error handling configuration.
        
        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            backoff_factor: Factor by which delay increases after each retry
            silent: If True, suppress all error messages
            default_return: Value to return on failure when max retries are exceeded
            allowed_exceptions: Tuple of exception types to catch (default: Exception)
            log_errors: Whether to log errors
            show_user_message: Whether to show error messages to the user
            user_message_prefix: Prefix for user-facing error messages
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.silent = silent
        self.default_return = default_return
        self.allowed_exceptions = allowed_exceptions or (Exception,)
        self.log_errors = log_errors
        self.show_user_message = show_user_message
        self.user_message_prefix = user_message_prefix


def with_error_handling(
    func: Optional[Callable[..., T]] = None,
    **config_kwargs
) -> Callable[..., Callable[..., T]]:
    """
    Decorator to add error handling to a function.
    
    Example:
        @with_error_handling(max_retries=3, default_return=None)
        def risky_operation():
            # Function implementation
            pass
    """
    config = ErrorHandlingConfig(**config_kwargs)
    
    def decorator(f: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            delay = config.initial_delay
            
            for attempt in range(config.max_retries + 1):
                try:
                    return f(*args, **kwargs)
                    
                except config.allowed_exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_retries:
                        break
                        
                    if config.log_errors and not config.silent:
                        logger.error(
                            f"Attempt {attempt + 1}/{config.max_retries} failed: {str(e)}",
                            exc_info=True
                        )
                    
                    if attempt < config.max_retries - 1:
                        time.sleep(delay)
                        delay = min(delay * config.backoff_factor, config.max_delay)
            
            # If we get here, all retries failed
            error_msg = f"{config.user_message_prefix}{str(last_exception) if last_exception else 'Unknown error'}"
            
            if config.log_errors and not config.silent:
                logger.error(f"All retries failed: {error_msg}", exc_info=last_exception)
            
            if config.show_user_message and not config.silent:
                st.error(error_msg)
            
            return config.default_return
            
        return wrapper
    
    if func is None:
        return decorator
    return decorator(func)


class ErrorBoundary:
    """
    Context manager for handling errors with fallback behavior.
    
    Example:
        with ErrorBoundary():
            # Code that might raise an exception
            result = risky_operation()
    """
    
    def __init__(
        self,
        default_return: Any = None,
        log_errors: bool = True,
        show_user_message: bool = True,
        user_message: str = "An error occurred",
        allowed_exceptions: Optional[tuple] = None
    ):
        """
        Initialize the error boundary.
        
        Args:
            default_return: Value to return if an exception occurs
            log_errors: Whether to log errors
            show_user_message: Whether to show error messages to the user
            user_message: Message to show to the user on error
            allowed_exceptions: Tuple of exception types to catch (default: Exception)
        """
        self.default_return = default_return
        self.log_errors = log_errors
        self.show_user_message = show_user_message
        self.user_message = user_message
        self.allowed_exceptions = allowed_exceptions or (Exception,)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is None:
            return True
            
        if not isinstance(exc_val, self.allowed_exceptions):
            return False
            
        if self.log_errors:
            logger.error(
                f"Error in error boundary: {str(exc_val)}",
                exc_info=(exc_type, exc_val, exc_tb)
            )
        
        if self.show_user_message:
            st.error(f"{self.user_message}: {str(exc_val)}")
        
        return True


def safe_call(
    func: Callable[..., T],
    *args: Any,
    default: Any = None,
    error_message: str = "An error occurred",
    log_errors: bool = True,
    show_user_message: bool = True,
    **kwargs: Any
) -> T:
    """
    Safely call a function with error handling.
    
    Args:
        func: Function to call
        *args: Positional arguments to pass to the function
        default: Value to return if the function raises an exception
        error_message: Error message to show/log on failure
        log_errors: Whether to log errors
        show_user_message: Whether to show error messages to the user
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        The result of the function call or the default value on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger.error(f"{error_message}: {str(e)}", exc_info=True)
        if show_user_message:
            st.error(f"{error_message}: {str(e)}")
        return default
