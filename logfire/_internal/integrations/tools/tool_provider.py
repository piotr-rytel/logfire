"""
This module provides functionality for instrumenting tool providers in agent frameworks.

It contains functions to create instrumented versions of tool methods, allowing for
detailed observability of tool calls made by agents.

Example usage:
    ```python
    from logfire import Logfire
    from logfire._internal.integrations.tools.tool_provider import instrument_tool_provider
    from .types import ToolConfig

    # Create tool configs
    tool_configs = [...]
    
    # Instrument a tool provider
    with instrument_tool_provider(logfire, tool_provider, tool_configs):
        # Use the instrumented tool provider
        result = tool_provider.some_tool(...)
    ```
"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import ExitStack, contextmanager, nullcontext
from typing import TYPE_CHECKING, Callable, ContextManager, cast, Optional
import types
import logging
from typing import Dict, List, TypeVar, Union

from ...utils import suppress_instrumentation

if TYPE_CHECKING:
    from ...main import Logfire
    from .types import ToolConfig


__all__ = ('instrument_tool_provider',)

# Set up a logger for debugging
logger = logging.getLogger("logfire.tools.provider")

# Constants for configuration
DEFAULT_TAG = "Tool"
MAX_RESULT_PREVIEW_LENGTH = 500

# Type definitions
T = TypeVar('T')
ToolProviderType = Union[object, type]


def instrument_tool_provider(
    logfire: 'Logfire',
    tool_provider: ToolProviderType,
    tool_configs: List['ToolConfig'],
) -> ContextManager[None]:
    """Instruments the provided `tool_provider` with `logfire`.

    The `tool_provider` argument can be:
    - a single tool provider instance
    - a class of a tool provider
    - an iterable of tool providers/classes

    Args:
        logfire: The Logfire instance to use for instrumentation
        tool_provider: The tool provider to instrument
        tool_configs: List of tool configurations

    Returns:
        A context manager that will revert the instrumentation when exited.
            Use of this context manager is optional.
    """
    # Validate inputs
    if logfire is None:
        raise ValueError("logfire instance cannot be None")
    if tool_configs is None or not tool_configs:
        raise ValueError("tool_configs cannot be None or empty")

    if isinstance(tool_provider, Iterable) and not isinstance(tool_provider, (str, bytes)):
        # Handle iterable of tool providers
        return _instrument_multiple_providers(logfire, tool_provider, tool_configs)

    if getattr(tool_provider, '_is_instrumented_by_logfire_tools', False):
        # Do nothing if already instrumented.
        return nullcontext()

    # Store original functions and replace with instrumented versions
    original_functions: Dict[str, Callable] = {}
    
    # Mark the tool provider as instrumented
    setattr(tool_provider, '_is_instrumented_by_logfire_tools', True)
    
    for tool_config in tool_configs:
        _instrument_single_tool(logfire, tool_provider, tool_config, original_functions)

    return _create_uninstrumentation_context(tool_provider, original_functions)


def _instrument_multiple_providers(
    logfire: 'Logfire',
    providers: Iterable[ToolProviderType],
    tool_configs: List['ToolConfig'],
) -> ContextManager[None]:
    """Instruments multiple tool providers.
    
    Args:
        logfire: The Logfire instance
        providers: Iterable of tool providers
        tool_configs: List of tool configurations
        
    Returns:
        A context manager for uninstrumentation
    """
    # Eagerly instrument each tool provider
    context_managers = [
        instrument_tool_provider(
            logfire,
            tp,
            tool_configs,
        )
        for tp in cast('Iterable[ToolProviderType]', providers)
    ]

    @contextmanager
    def uninstrument_context():
        with ExitStack() as exit_stack:
            for context_manager in context_managers:
                exit_stack.enter_context(context_manager)
            yield

    return uninstrument_context()


def _instrument_single_tool(
    logfire: 'Logfire',
    tool_provider: ToolProviderType,
    tool_config: 'ToolConfig',
    original_functions: Dict[str, Callable],
) -> None:
    """Instruments a single tool on a provider.
    
    Args:
        logfire: The Logfire instance
        tool_provider: The tool provider to instrument
        tool_config: Configuration for the tool
        original_functions: Dictionary to store original functions
    """
    try:
        tool_name = tool_config.name
        
        # Verify the tool exists on the provider
        if not hasattr(tool_provider, tool_name):
            logger.warning(f"Tool {tool_name} not found on {tool_provider.__class__.__name__}")
            return
            
        # Get and store the original function
        original_function = getattr(tool_provider, tool_name)
        original_functions[tool_name] = original_function
        
        # Get the instrumented function
        instrumented_function = tool_config.instrumented_function
        
        # Handle instance methods vs class methods differently
        is_instance = _is_instance_not_class(tool_provider)
        if is_instance:
            _instrument_instance_method(logfire, tool_provider, tool_name, instrumented_function, tool_config)
        else:
            # For class methods, just replace the function
            setattr(tool_provider, tool_name, instrumented_function)
                
    except AttributeError as e:
        logger.error(f"Failed to set {tool_config.name} on {tool_provider.__class__.__name__}: {e}")
    except TypeError as e:
        logger.error(f"Type error instrumenting {tool_config.name}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error instrumenting {tool_config.name}: {e}", exc_info=True)


def _is_instance_not_class(obj: ToolProviderType) -> bool:
    """Determines if an object is an instance and not a class.
    
    Args:
        obj: The object to check
        
    Returns:
        True if the object is an instance, False if it's a class
    """
    return isinstance(obj, object) and not isinstance(obj, type)


def _instrument_instance_method(
    logfire: 'Logfire',
    tool_provider: ToolProviderType,
    tool_name: str,
    instrumented_function: Callable,
    tool_config: 'ToolConfig',
) -> None:
    """Instruments an instance method on a tool provider.
    
    Args:
        logfire: The Logfire instance
        tool_provider: The tool provider instance
        tool_name: Name of the tool method
        instrumented_function: The instrumented function
        tool_config: Configuration for the tool
    """
    logger.debug(f"Instrumenting instance method {tool_name} on {tool_provider.__class__.__name__}")
    
    # Create a bound method
    bound_method = types.MethodType(instrumented_function, tool_provider)
    
    # Replace the method on the instance
    setattr(tool_provider, tool_name, bound_method)
    
    # Handle function objects if present
    _instrument_function_object(logfire, tool_provider, tool_name, tool_config)


def _instrument_function_object(
    logfire: 'Logfire',
    tool_provider: ToolProviderType,
    tool_name: str,
    tool_config: 'ToolConfig',
) -> None:
    """Instruments a function object if it exists.
    
    Args:
        logfire: The Logfire instance
        tool_provider: The tool provider
        tool_name: Name of the tool
        tool_config: Configuration for the tool
    """
    # If the tool provider has a functions attribute (for Toolkit class)
    if hasattr(tool_provider, 'functions') and tool_name in tool_provider.functions:
        # Get the function object
        function_obj = tool_provider.functions[tool_name]
        
        # If we have an instrument_function, use it to instrument the function object
        if tool_config.instrument_function:
            logger.debug(f"Instrumenting function object {tool_name} on {tool_provider.__class__.__name__}")
            try:
                tool_config.instrument_function(function_obj)
            except Exception as e:
                logger.error(f"Failed to instrument function object {tool_name}: {e}")


def _create_uninstrumentation_context(
    tool_provider: ToolProviderType,
    original_functions: Dict[str, Callable],
) -> ContextManager[None]:
    """Creates a context manager for uninstrumentation.
    
    Args:
        tool_provider: The tool provider
        original_functions: Dictionary of original functions
        
    Returns:
        A context manager that will uninstrument when exited
    """
    @contextmanager
    def uninstrument_context():
        """Context manager to remove instrumentation from the tool provider.

        The user isn't required (or even expected) to use this context manager,
        which is why the instrumenting has already happened before.
        It exists mostly for tests and just in case users want it.
        """
        try:
            yield
        finally:
            _uninstrument_provider(tool_provider, original_functions)

    return uninstrument_context()


def _uninstrument_provider(
    tool_provider: ToolProviderType,
    original_functions: Dict[str, Callable],
) -> None:
    """Uninstruments a tool provider.
    
    Args:
        tool_provider: The tool provider to uninstrument
        original_functions: Dictionary of original functions
    """
    # Restore original functions
    for tool_name, original_function in original_functions.items():
        try:
            setattr(tool_provider, tool_name, original_function)
        except Exception as e:
            logger.error(f"Failed to restore original function {tool_name}: {e}")
    
    # Remove the instrumentation flag
    try:
        setattr(tool_provider, '_is_instrumented_by_logfire_tools', False)
    except Exception as e:
        logger.error(f"Failed to remove instrumentation flag: {e}")


@contextmanager
def maybe_suppress_instrumentation(suppress: bool) -> None:
    """Conditionally suppresses instrumentation.
    
    Args:
        suppress: Whether to suppress instrumentation
    """
    if suppress:
        with suppress_instrumentation():
            yield
    else:
        yield 


def create_instrumented_method(
    logfire: 'Logfire',
    original_method: Callable,
    method_name: str,
    tags: Optional[List[str]] = None
) -> Callable:
    """
    Create an instrumented version of a tool method.
    
    This function wraps the original method with instrumentation that creates
    spans for each method call, captures arguments and results, and logs relevant
    information for debugging and observability.
    
    Args:
        logfire: The Logfire instance to use for instrumentation
        original_method: The original method to instrument
        method_name: The name of the method
        tags: Optional list of tags to apply to spans
        
    Returns:
        A wrapped version of the method with added instrumentation
    """
    def wrapper(*args, **kwargs):
        # Extract the query parameter if it exists
        query = kwargs.get('query', args[1] if len(args) > 1 else 'unknown')
        
        # Use default tags if none provided
        method_tags = tags or [DEFAULT_TAG]
        
        # Create a span for the method call
        with logfire.span(f"Tool Call: {method_name}", 
                 query=query, _tags=method_tags) as span:
            
            # Add parameters to the span
            for k, v in kwargs.items():
                span.set_attribute(f"{method_name}.{k}", str(v))
            
            try:
                # Call the original method
                result = original_method(*args, **kwargs)
                
                # Add result information to span
                if result:
                    try:
                        span.set_attribute(
                            f"{method_name}.result_preview", 
                            str(result)[:MAX_RESULT_PREVIEW_LENGTH]
                        )
                    except Exception as e:
                        logger.debug(
                            f"Error adding result preview to span for {method_name}: {e}",
                            exc_info=True
                        )
                
                return result
            except Exception as e:
                # Log the exception and add it to the span
                span.set_attribute(f"{method_name}.error", str(e))
                logger.error(f"Error in tool method {method_name}: {e}", exc_info=True)
                raise
    
    return wrapper


def instrument_tools(
    logfire: 'Logfire',
    tools: Optional[List[object]] = None,
    tags: Optional[List[str]] = None,
    **kwargs
) -> None:
    """
    Instrument a list of tool instances for observability.
    
    This function adds instrumentation to tool instances, creating spans for each
    tool call and capturing relevant information for debugging and observability.
    
    Args:
        logfire: The Logfire instance to use for instrumentation
        tools: A list of tool instances to instrument
        tags: Optional list of tags to apply to spans
        **kwargs: Additional keyword arguments for backward compatibility
    """
    # Handle the case where tools is passed as a keyword argument
    if tools is None and 'tools' in kwargs:
        tools = kwargs.get('tools')
    
    if not tools:
        logger.warning("No tools provided for instrumentation")
        return
        
    # Set default tags
    method_tags = tags or [DEFAULT_TAG]
    
    # Import tool providers
    from logfire._internal.integrations.tools.duckduckgo import get_duckduckgo_tool_configs
    
    # Map of tool class names to provider functions
    # This could be made extensible through a registration mechanism
    providers = {
        "DuckDuckGoTools": get_duckduckgo_tool_configs
    }
    
    # Process each tool
    for tool in tools:
        if tool is None:
            continue
            
        # Get the class name of the tool
        tool_class_name = tool.__class__.__name__
        
        # Check if we have a provider for this tool
        if tool_class_name in providers:
            provider_func = providers[tool_class_name]
            
            try:
                # Get the tool configs
                tool_configs = provider_func()
                
                # Instrument the tool
                instrument_tool(logfire, tool, tool_configs, method_tags)
                
                # Also instrument the class to handle class methods
                instrument_tool(logfire, tool.__class__, tool_configs, method_tags)
                
                logger.debug(f"Instrumented {tool_class_name}")
            except Exception as e:
                logger.error(f"Failed to instrument {tool_class_name}: {e}", exc_info=True)
        else:
            logger.warning(f"No instrumentation available for {tool_class_name}")


def instrument_tool(
    logfire: 'Logfire',
    tool: ToolProviderType,
    tool_configs: List['ToolConfig'],
    tags: Optional[List[str]] = None
) -> None:
    """
    Instrument a tool instance or class for observability.
    
    This function adds instrumentation to a tool instance or class, creating spans
    for each tool call and capturing relevant information for debugging and observability.
    
    Args:
        logfire: The Logfire instance to use for instrumentation
        tool: The tool instance or class to instrument
        tool_configs: A list of ToolConfig objects for the tool
        tags: Optional list of tags to apply to spans
    """
    if tool is None:
        logger.warning("Cannot instrument None tool")
        return
        
    if tool_configs is None or not tool_configs:
        logger.warning(f"No tool configs provided for {tool.__class__.__name__}")
        return
        
    # Set default tags
    method_tags = tags or [DEFAULT_TAG]
    
    # Process each tool config
    for config in tool_configs:
        if not config:
            continue
            
        method_name = config.method_name
        
        try:
            # Check if the method exists on the tool
            if hasattr(tool, method_name):
                # Get the original method
                original_method = getattr(tool, method_name)
                
                # Create the instrumented method with the logfire instance
                wrapped_method = create_instrumented_method(
                    logfire, 
                    original_method, 
                    method_name, 
                    config.tags or method_tags
                )
                
                # Replace the original method with the instrumented one
                setattr(tool, method_name, wrapped_method)
                
                logger.debug(f"Instrumented {method_name} on {tool.__class__.__name__}")
                
                # Check if we need to instrument the Function object
                if config.instrument_function and hasattr(tool, "functions"):
                    _instrument_function_in_dict(tool, method_name, config)
            else:
                logger.debug(f"Method {method_name} not found on {tool.__class__.__name__}")
        except AttributeError as e:
            logger.error(f"Attribute error instrumenting {method_name}: {e}")
        except TypeError as e:
            logger.error(f"Type error instrumenting {method_name}: {e}")
        except Exception as e:
            logger.error(f"Failed to instrument {method_name}: {e}", exc_info=True)


def _instrument_function_in_dict(
    tool: ToolProviderType,
    method_name: str,
    config: 'ToolConfig'
) -> None:
    """
    Instruments a function object in a dictionary.
    
    Args:
        tool: The tool containing the functions dictionary
        method_name: The name of the method/function
        config: The tool configuration
    """
    # Get the functions dictionary
    functions = getattr(tool, "functions", {})
    
    # Check if the method name is in the functions dictionary
    if method_name in functions:
        # Get the function object
        function_obj = functions[method_name]
        
        try:
            # Instrument the function object
            config.instrument_function(function_obj)
            logger.debug(f"Instrumented Function object for {method_name}")
        except Exception as e:
            logger.error(f"Failed to instrument Function object for {method_name}: {e}") 