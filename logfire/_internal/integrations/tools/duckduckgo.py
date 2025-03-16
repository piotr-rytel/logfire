"""
This module provides instrumentation for DuckDuckGo tools used in agent frameworks.

It contains functions to create instrumented versions of DuckDuckGo search and news methods,
allowing for detailed observability of DuckDuckGo tool calls made by agents.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, List, TypeVar

import logfire
from logfire._internal.integrations.tools.types import ToolConfig

logger = logging.getLogger(__name__)

# Type variable for the return type of the wrapped function
T = TypeVar('T')

# Default preview length for result logging
DEFAULT_PREVIEW_LENGTH = 500


def _truncate_preview(text: str, max_length: int = DEFAULT_PREVIEW_LENGTH) -> str:
    """
    Truncate text to max_length and add indicator if truncated.
    
    Args:
        text: The text to truncate
        max_length: Maximum length before truncation
        
    Returns:
        Truncated text with indicator if needed
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "... [truncated]"


def _process_result(result: Any, span: logfire.Span, attribute_prefix: str) -> None:
    """
    Process and attach the result to the span.
    
    Args:
        result: The result to process
        span: The span to attach the result to
        attribute_prefix: Prefix for the span attribute name
    """
    try:
        if isinstance(result, str):
            try:
                result_data = json.loads(result)
                # Store both the full result and a summary
                if isinstance(result_data, dict):
                    span.set_attribute(f"{attribute_prefix}.result_keys", list(result_data.keys()))
                    span.set_attribute(f"{attribute_prefix}.result_data", result_data)
                elif isinstance(result_data, list):
                    span.set_attribute(f"{attribute_prefix}.result_length", len(result_data))
                    span.set_attribute(f"{attribute_prefix}.result_data", result_data)
                    # Add a preview of the first few items
                    if result_data:
                        preview_items = result_data[:3] if len(result_data) > 3 else result_data
                        span.set_attribute(f"{attribute_prefix}.result_preview", preview_items)
            except json.JSONDecodeError:
                # If it's not valid JSON, just log the first part of the result
                span.set_attribute(f"{attribute_prefix}.result_preview", _truncate_preview(result))
        elif isinstance(result, list):
            # For lists, store length and a preview of the first few items
            span.set_attribute(f"{attribute_prefix}.result_length", len(result))
            if result:
                preview_items = result[:3] if len(result) > 3 else result
                span.set_attribute(f"{attribute_prefix}.result_preview", preview_items)
                span.set_attribute(f"{attribute_prefix}.result_data", result)
        elif isinstance(result, dict):
            # For dictionaries, store keys and the full data
            span.set_attribute(f"{attribute_prefix}.result_keys", list(result.keys()))
            span.set_attribute(f"{attribute_prefix}.result_data", result)
        else:
            # For other types, convert to string and truncate
            span.set_attribute(f"{attribute_prefix}.result_preview", _truncate_preview(str(result)))
    except Exception as e:
        logger.debug(f"Error processing {attribute_prefix} result: {e}")
        # Log the error but still try to capture something about the result
        try:
            span.set_attribute(f"{attribute_prefix}.result_type", str(type(result)))
            if result is not None:
                span.set_attribute(f"{attribute_prefix}.result_preview", _truncate_preview(str(result)))
        except (AttributeError, TypeError, ValueError) as err:
            logger.debug(f"Failed to set fallback attributes: {err}")


def create_instrumented_duckduckgo_method(
    original_func: Callable[..., T], 
    span_name: str, 
    attribute_prefix: str,
    tags: List[str],
    preview_length: int = DEFAULT_PREVIEW_LENGTH
) -> Callable[..., T]:
    """
    Create an instrumented version of a DuckDuckGo method.
    
    This function creates a wrapper that adds instrumentation to DuckDuckGo methods,
    creating spans for each call, capturing queries and results, and logging relevant
    information for debugging and observability.
    
    Args:
        original_func: The original method to instrument
        span_name: The name to use for the span
        attribute_prefix: Prefix for span attributes
        tags: Tags to apply to the span
        preview_length: Length to truncate previews (defaults to DEFAULT_PREVIEW_LENGTH)
        
    Returns:
        A wrapped version of the method with added instrumentation
        
    Raises:
        TypeError: If original_func is not callable
    """
    if not callable(original_func):
        raise TypeError("original_func must be callable")
        
    def wrapper(*args: Any, **kwargs: Any) -> T:
        # Extract self and query
        self = args[0]
        query = kwargs.get('query', None)
        if query is None and len(args) > 1:
            query = args[1]
        
        # Default to "unknown" if query is None or empty
        if not query:
            query = "unknown"
        
        with logfire.span(span_name, query=query, _tags=["Tool", "DuckDuckGo"] + tags) as span:
            try:
                result = original_func(*args, **kwargs)
                # Log the raw result type for debugging
                span.set_attribute(f"{attribute_prefix}.result_type", str(type(result)))
                
                if result is not None:
                    _process_result(result, span, attribute_prefix)
                    
                    # Check if we already have a preview attribute
                    attributes = getattr(span, 'attributes', {})
                    has_preview = f"{attribute_prefix}.result_preview" in attributes
                    
                    # Add a summary if we don't have a preview and result is complex
                    if isinstance(result, (list, dict)) and not has_preview:
                        span.set_attribute(
                            f"{attribute_prefix}.result_summary", 
                            _truncate_preview(str(result), preview_length)
                        )
                else:
                    span.set_attribute(f"{attribute_prefix}.result", "None")
                
                return result
            except Exception as e:
                span.record_exception(e)
                logger.error(f"Error in {span_name} with query '{query}': {e}")
                raise
    
    return wrapper


def instrumented_duckduckgo_search(
    original_func: Callable[..., T], 
    preview_length: int = DEFAULT_PREVIEW_LENGTH
) -> Callable[..., T]:
    """
    Create an instrumented version of the DuckDuckGo search method.
    
    This function wraps the original search method with instrumentation that creates
    spans for each search call, captures the query and results, and logs relevant
    information for debugging and observability.
    
    Args:
        original_func: The original search method to instrument
        preview_length: Length to truncate previews (defaults to DEFAULT_PREVIEW_LENGTH)
        
    Returns:
        A wrapped version of the search method with added instrumentation
    
    Example:
        ```python
        from duckduckgo_search import DDGS
        import logfire
        
        # Initialize logfire
        logfire.init()
        
        # Instrument the search method
        from logfire._internal.integrations.tools.duckduckgo import instrumented_duckduckgo_search
        DDGS.text = instrumented_duckduckgo_search(DDGS.text)
        
        # Use the instrumented method
        ddgs = DDGS()
        results = ddgs.text("python programming")
        ```
    """
    return create_instrumented_duckduckgo_method(
        original_func, 
        "DuckDuckGo Search", 
        "duckduckgo_search",
        ["Search"],
        preview_length
    )


def instrumented_duckduckgo_news(
    original_func: Callable[..., T],
    preview_length: int = DEFAULT_PREVIEW_LENGTH
) -> Callable[..., T]:
    """
    Create an instrumented version of the DuckDuckGo news method.
    
    This function wraps the original news method with instrumentation that creates
    spans for each news call, captures the query and results, and logs relevant
    information for debugging and observability.
    
    Args:
        original_func: The original news method to instrument
        preview_length: Length to truncate previews (defaults to DEFAULT_PREVIEW_LENGTH)
        
    Returns:
        A wrapped version of the news method with added instrumentation
    
    Example:
        ```python
        from duckduckgo_search import DDGS
        import logfire
        
        # Initialize logfire
        logfire.init()
        
        # Instrument the news method
        from logfire._internal.integrations.tools.duckduckgo import instrumented_duckduckgo_news
        DDGS.news = instrumented_duckduckgo_news(DDGS.news)
        
        # Use the instrumented method
        ddgs = DDGS()
        results = ddgs.news("latest technology")
        ```
    """
    return create_instrumented_duckduckgo_method(
        original_func, 
        "DuckDuckGo News", 
        "duckduckgo_news",
        ["News"],
        preview_length
    )


def get_duckduckgo_tool_configs(preview_length: int = DEFAULT_PREVIEW_LENGTH) -> List[ToolConfig]:
    """
    Get the tool configurations for DuckDuckGo tools.
    
    This function returns a list of ToolConfig objects that can be used to instrument
    DuckDuckGo tools in agent frameworks.
    
    Args:
        preview_length: Custom preview length for result truncation
                       (defaults to DEFAULT_PREVIEW_LENGTH)
    
    Returns:
        A list of ToolConfig objects for DuckDuckGo tools
    
    Example:
        ```python
        import logfire
        from logfire._internal.integrations.tools.duckduckgo import get_duckduckgo_tool_configs
        from logfire._internal.integrations.tools.instrumentor import instrument_tools
        
        # Initialize logfire
        logfire.init()
        
        # Get the tool configs
        tool_configs = get_duckduckgo_tool_configs()
        
        # Instrument the tools in an agent framework
        instrument_tools(agent_tools, tool_configs)
        ```
    """
    # Create closures that capture the preview_length
    def search_instrumentor(func: Callable[..., T]) -> Callable[..., T]:
        return instrumented_duckduckgo_search(func, preview_length)
    
    def news_instrumentor(func: Callable[..., T]) -> Callable[..., T]:
        return instrumented_duckduckgo_news(func, preview_length)
    
    return [
        ToolConfig(
            method_name="duckduckgo_search",
            instrumented_method=search_instrumentor,
            tags=["DuckDuckGo", "Search"]
        ),
        ToolConfig(
            method_name="duckduckgo_news",
            instrumented_method=news_instrumentor,
            tags=["DuckDuckGo", "News"]
        )
    ] 