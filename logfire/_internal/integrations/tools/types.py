"""
This module defines the types used for tool instrumentation in Logfire.

It contains dataclasses that represent configuration for different types of tools
and instrumentation approaches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, List


@dataclass
class ToolConfig:
    """
    Configuration for instrumenting a tool method.
    
    This class holds the configuration needed to instrument a method on a tool instance,
    including the method name, the instrumented version of the method, and any tags
    to apply to the spans created during instrumentation.
    
    Attributes:
        method_name: The name of the method to instrument
        instrumented_method: The instrumented version of the method
        tags: Optional list of tags to apply to spans
        instrument_function: Optional function to instrument a Function object
    """
    method_name: str
    instrumented_method: Callable
    tags: Optional[List[str]] = None
    instrument_function: Optional[Callable] = None


@dataclass
class FunctionCallConfig:
    """
    Configuration for instrumenting a FunctionCall class.
    
    This class holds the configuration needed to instrument a FunctionCall class,
    which is used by agent frameworks to execute tool calls.
    
    Attributes:
        function_call_class: The FunctionCall class to instrument
        tags: Optional list of tags to apply to spans
    """
    function_call_class: Any
    tags: Optional[List[str]] = None 