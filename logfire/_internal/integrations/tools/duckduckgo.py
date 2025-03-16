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
                span.set_attribute(f"{attribute_prefix}.result_data", result_data)
            except json.JSONDecodeError:
                # If it's not valid JSON, just log the first part of the result
                span.set_attribute(f"{attribute_prefix}.result_preview", result[:DEFAULT_PREVIEW_LENGTH])
        elif isinstance(result, (list, dict)):
            span.set_attribute(f"{attribute_prefix}.result_data", result)
        else:
            # For other types, convert to string and truncate
            span.set_attribute(f"{attribute_prefix}.result_preview", str(result)[:DEFAULT_PREVIEW_LENGTH])
    except Exception as e:
        logger.debug(f"Error processing {attribute_prefix} result: {e}")


def create_instrumented_duckduckgo_method(
    original_func: Callable[..., T], 
    span_name: str, 
    attribute_prefix: str,
    tags: List[str]
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
        
    Returns:
        A wrapped version of the method with added instrumentation
    """
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
                if result is not None:
                    _process_result(result, span, attribute_prefix)
                return result
            except Exception as e:
                span.record_exception(e)
                logger.error(f"Error in {span_name} with query '{query}': {e}")
                raise
    
    return wrapper


def instrumented_duckduckgo_search(original_func: Callable[..., T]) -> Callable[..., T]:
    """
    Create an instrumented version of the DuckDuckGo search method.
    
    This function wraps the original search method with instrumentation that creates
    spans for each search call, captures the query and results, and logs relevant
    information for debugging and observability.
    
    Args:
        original_func: The original search method to instrument
        
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
        ["Search"]
    )


def instrumented_duckduckgo_news(original_func: Callable[..., T]) -> Callable[..., T]:
    """
    Create an instrumented version of the DuckDuckGo news method.
    
    This function wraps the original news method with instrumentation that creates
    spans for each news call, captures the query and results, and logs relevant
    information for debugging and observability.
    
    Args:
        original_func: The original news method to instrument
        
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
        ["News"]
    )


def get_duckduckgo_tool_configs(preview_length: int = DEFAULT_PREVIEW_LENGTH) -> List[ToolConfig]:
    """
    Get the tool configurations for DuckDuckGo tools.
    
    This function returns a list of ToolConfig objects that can be used to instrument
    DuckDuckGo tools in agent frameworks.
    
    Args:
        preview_length: Optional custom preview length for result truncation
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
    # Set the module-level preview length if a custom one is provided
    global DEFAULT_PREVIEW_LENGTH
    if preview_length != DEFAULT_PREVIEW_LENGTH:
        DEFAULT_PREVIEW_LENGTH = preview_length
    
    return [
        ToolConfig(
            method_name="duckduckgo_search",
            instrumented_method=instrumented_duckduckgo_search,
            tags=["DuckDuckGo", "Search"]
        ),
        ToolConfig(
            method_name="duckduckgo_news",
            instrumented_method=instrumented_duckduckgo_news,
            tags=["DuckDuckGo", "News"]
        )
    ] 