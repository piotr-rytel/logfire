"""
This module provides functionality for instrumenting FunctionCall objects in agent frameworks.

It contains functions to create instrumented versions of the FunctionCall.execute method,
allowing for detailed observability of tool calls made by agents.
"""

from __future__ import annotations

import json
import logging
import functools
import traceback
from typing import Any, Callable, TypeVar, Protocol, Optional, List, Dict

import logfire

logger = logging.getLogger(__name__)

# Type definitions to improve type annotations
T = TypeVar('T')

# Define a Protocol for span objects to improve type annotations
class SpanProtocol(Protocol):
    """Protocol defining the required interface for a span object."""
    def set_attribute(self, key: str, value: Any) -> None: ...
    
# Define a Protocol for FunctionCallClass to make the type more specific
class FunctionCallProtocol(Protocol):
    """Protocol defining the required interface for a FunctionCall class."""
    def execute(self) -> Any: ...
    
FunctionCallClass = TypeVar('FunctionCallClass', bound=FunctionCallProtocol)

# Constants
MAX_PREVIEW_LENGTH = 500
IMPORTANT_KEYS = ['title', 'url', 'body', 'snippet', 'text', 'content', 'description']
SENSITIVE_PARAM_NAMES = ['password', 'token', 'secret', 'key', 'auth', 'credential']


def _truncate_string(s: str, max_length: int = MAX_PREVIEW_LENGTH) -> str:
    """
    Truncate a string to the specified maximum length and add indicator if truncated.
    
    Args:
        s: The string to truncate
        max_length: Maximum length of the string
        
    Returns:
        The truncated string with indicator if truncated
    """
    if len(s) <= max_length:
        return s
    return s[:max_length] + "... [truncated]"


def _is_sensitive_param(param_name: str) -> bool:
    """
    Check if a parameter name might contain sensitive information.
    
    Args:
        param_name: The name of the parameter to check
        
    Returns:
        True if the parameter might contain sensitive information, False otherwise
    """
    return any(sensitive in param_name.lower() for sensitive in SENSITIVE_PARAM_NAMES)


def _has_attribute(span_obj: SpanProtocol, attr_name: str) -> bool:
    """
    Safely check if a span object has an attribute.
    
    Args:
        span_obj: The span object to check
        attr_name: The name of the attribute to check for
        
    Returns:
        True if the attribute exists, False otherwise
    """
    try:
        if hasattr(span_obj, '_get_attribute'):
            return span_obj._get_attribute(attr_name, None) is not None
        return attr_name in getattr(span_obj, 'attributes', {})
    except Exception:
        return False


def _process_string_result(function_name: str, result: str, span_obj: SpanProtocol) -> None:
    """
    Process a string result and add relevant attributes to the span.
    
    Args:
        function_name: The name of the function that was called
        result: The string result to process
        span_obj: The span object to add attributes to
    """
    preview = _truncate_string(result)
    span_obj.set_attribute(f"{function_name}.result_preview", preview)


def _process_list_result(function_name: str, result: List[Any], span_obj: SpanProtocol) -> None:
    """
    Process a list result and add relevant attributes to the span.
    
    Args:
        function_name: The name of the function that was called
        result: The list result to process
        span_obj: The span object to add attributes to
    """
    span_obj.set_attribute(f"{function_name}.result_length", len(result))
    if result:
        preview_items = result[:3] if len(result) > 3 else result
        span_obj.set_attribute(f"{function_name}.result_preview", str(preview_items))
        span_obj.set_attribute(f"{function_name}.result_data", result)


def _process_dict_result(function_name: str, result: Dict[str, Any], span_obj: SpanProtocol) -> None:
    """
    Process a dictionary result and add relevant attributes to the span.
    
    Args:
        function_name: The name of the function that was called
        result: The dictionary result to process
        span_obj: The span object to add attributes to
    """
    span_obj.set_attribute(f"{function_name}.result_keys", list(result.keys()))
    
    # Store a preview of the values for important keys
    for key in result.keys():
        if key in IMPORTANT_KEYS:
            value = result[key]
            if isinstance(value, str):
                preview = _truncate_string(value)
                span_obj.set_attribute(f"{function_name}.result.{key}", preview)
    
    span_obj.set_attribute(f"{function_name}.result_data", result)


def _process_json_result(function_name: str, result: str, span_obj: SpanProtocol) -> None:
    """
    Process a JSON string result and add relevant attributes to the span.
    
    Args:
        function_name: The name of the function that was called
        result: The JSON string result to process
        span_obj: The span object to add attributes to
    """
    try:
        result_data = json.loads(result)
        
        if isinstance(result_data, dict):
            _process_dict_result(function_name, result_data, span_obj)
        elif isinstance(result_data, list):
            _process_list_result(function_name, result_data, span_obj)
        
        # Store the full result data for detailed inspection
        span_obj.set_attribute(f"{function_name}.result_data", result_data)
    except json.JSONDecodeError:
        # If JSON parsing fails, log the preview
        _process_string_result(function_name, result, span_obj)


def _capture_fallback_result(function_name: str, result: Any, span_obj: SpanProtocol) -> None:
    """
    Capture a fallback result when other processing methods fail.
    
    Args:
        function_name: The name of the function that was called
        result: The result to capture
        span_obj: The span object to add attributes to
    """
    try:
        if result is not None:
            preview = _truncate_string(str(result))
            span_obj.set_attribute(f"{function_name}.result_preview", preview)
    except Exception as e:
        logger.debug(f"Failed to capture fallback result: {e}")


def _process_result(function_name: str, result: Any, span_obj: SpanProtocol) -> None:
    """
    Process the result of a function call and add relevant attributes to the span.
    
    Args:
        function_name: The name of the function that was called
        result: The result of the function call
        span_obj: The span object to add attributes to
    """
    try:
        # Always store the result type for debugging
        span_obj.set_attribute(f"{function_name}.result_type", str(type(result)))
        
        if isinstance(result, str) and (result.startswith('{') or result.startswith('[')):
            _process_json_result(function_name, result, span_obj)
        elif isinstance(result, str):
            _process_string_result(function_name, result, span_obj)
        elif isinstance(result, list):
            _process_list_result(function_name, result, span_obj)
        elif isinstance(result, dict):
            _process_dict_result(function_name, result, span_obj)
    except Exception as e:
        logger.debug(
            f"Error processing {function_name} result: {e}\n"
            f"{traceback.format_exc()}"
        )
        _capture_fallback_result(function_name, result, span_obj)


def create_instrumented_execute(original_execute: Callable[..., T], tags: Optional[List[str]] = None) -> Callable[..., T]:
    """
    Create an instrumented version of the FunctionCall.execute method.
    
    This function wraps the original execute method with instrumentation that creates
    spans for each function call, captures arguments and results, and logs relevant
    information for debugging and observability.
    
    Args:
        original_execute: The original execute method to instrument
        tags: Optional list of tags to include in the span
        
    Returns:
        A wrapped version of the execute method with added instrumentation
        
    Example:
        ```python
        from my_agent_framework import FunctionCall
        
        # Instrument the FunctionCall class
        original_execute = FunctionCall.execute
        FunctionCall.execute = create_instrumented_execute(original_execute)
        ```
    """
    tags = tags or ["Tool"]
    
    @functools.wraps(original_execute)
    def instrumented_execute(self):
        # Get the function name and arguments
        function_name = getattr(self.function, 'name', 'unknown_function')
        arguments = getattr(self, 'arguments', {}) or {}
        
        # Get the query parameter if it exists, ensure it's string-like
        query = str(arguments.get('query', 'unknown'))
        
        # Create a span for the function call
        with logfire.span(f"Tool {function_name} query: {query}", 
                 query=query, _tags=tags + ['FunctionCall']) as span_obj:
            
            # Add parameters to the span
            for k, v in arguments.items():
                # Skip sensitive parameters
                if _is_sensitive_param(k):
                    span_obj.set_attribute(f"{function_name}.param.{k}", "[REDACTED]")
                # Avoid storing potentially large values
                elif isinstance(v, (str, int, float, bool)) or v is None:
                    span_obj.set_attribute(f"{function_name}.param.{k}", v)
                else:
                    span_obj.set_attribute(f"{function_name}.param.{k}.type", str(type(v)))
            
            try:
                # Call the original execute method
                result = original_execute(self)
                logger.debug(f"Successfully executed tool {function_name}")
                
                # Add result information to span
                if hasattr(self, 'result') and self.result:
                    _process_result(function_name, self.result, span_obj)
                    
                    # Check if we need to add a basic representation of the result
                    has_preview = _has_attribute(span_obj, f"{function_name}.result_preview")
                    has_data = _has_attribute(span_obj, f"{function_name}.result_data")
                    
                    if not has_preview and not has_data:
                        preview = _truncate_string(str(self.result))
                        span_obj.set_attribute(f"{function_name}.result_summary", preview)
                
                return result
            except Exception as e:
                logger.error(
                    f"Error executing tool {function_name}: {e}\n"
                    f"{traceback.format_exc()}"
                )
                span_obj.set_attribute(f"{function_name}.error", str(e))
                span_obj.set_attribute(f"{function_name}.error_type", type(e).__name__)
                raise
    
    return instrumented_execute


def instrument_function_call(function_call_class: FunctionCallClass, tags: Optional[List[str]] = None) -> FunctionCallClass:
    """
    Instrument a FunctionCall class to add observability.
    
    This function replaces the execute method of the provided FunctionCall class
    with an instrumented version that creates spans and logs information about
    each function call.
    
    Args:
        function_call_class: The FunctionCall class to instrument
        tags: Optional list of tags to include in the span
        
    Returns:
        The instrumented FunctionCall class
        
    Example:
        ```python
        from my_agent_framework import FunctionCall
        
        # Instrument the FunctionCall class
        instrumented_class = instrument_function_call(FunctionCall)
        ```
        
    Raises:
        AttributeError: If the provided class doesn't have an execute method
    """
    # Validate that the class has an execute method
    if not hasattr(function_call_class, 'execute'):
        raise AttributeError(f"Class {function_call_class.__name__} has no 'execute' method")
    
    # Check if already instrumented
    if getattr(function_call_class, '_is_instrumented_by_logfire', False):
        logger.debug(f"FunctionCall class already instrumented: {function_call_class.__name__}")
        return function_call_class
    
    # Store the original execute method
    original_execute = function_call_class.execute
    
    # Create an instrumented version of the execute method
    instrumented_execute = create_instrumented_execute(original_execute, tags)
    
    # Replace the execute method with our instrumented version
    function_call_class.execute = instrumented_execute
    
    # Mark the class as instrumented
    function_call_class._is_instrumented_by_logfire = True
    function_call_class._original_execute = original_execute
    
    logger.info(f"Successfully instrumented FunctionCall class: {function_call_class.__name__}")
    
    # Return the instrumented class
    return function_call_class


def uninstrument_function_call(function_call_class: FunctionCallClass) -> FunctionCallClass:
    """
    Removes instrumentation from the FunctionCall class.
    
    Args:
        function_call_class: The FunctionCall class to uninstrument.
        
    Returns:
        The uninstrumented FunctionCall class
        
    Example:
        ```python
        from my_agent_framework import FunctionCall
        
        # Uninstrument the FunctionCall class
        uninstrumented_class = uninstrument_function_call(FunctionCall)
        ```
    """
    if not getattr(function_call_class, '_is_instrumented_by_logfire', False):
        logger.debug(f"FunctionCall class not instrumented: {function_call_class.__name__}")
        return function_call_class
    
    try:
        # Restore the original execute method
        function_call_class.execute = function_call_class._original_execute
        
        # Remove the instrumentation flag and original execute reference
        delattr(function_call_class, '_is_instrumented_by_logfire')
        delattr(function_call_class, '_original_execute')
        
        logger.info(f"Successfully uninstrumented FunctionCall class: {function_call_class.__name__}")
    except AttributeError as e:
        logger.error(
            f"Error uninstrumenting FunctionCall class {function_call_class.__name__}: {e}\n"
            f"{traceback.format_exc()}"
        )
    
    return function_call_class 