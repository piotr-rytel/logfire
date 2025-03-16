"""**Logfire** is the observability tool focused on developer experience."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any

from logfire.sampling import SamplingOptions

from ._internal.auto_trace import AutoTraceModule
from ._internal.auto_trace.rewrite_ast import no_auto_trace
from ._internal.cli import logfire_info
from ._internal.config import AdvancedOptions, CodeSource, ConsoleOptions, MetricsOptions, PydanticPlugin, configure
from ._internal.constants import LevelName
from ._internal.main import Logfire, LogfireSpan
from ._internal.scrubbing import ScrubbingOptions, ScrubMatch
from ._internal.stack_info import add_non_user_code_prefix
from ._internal.utils import suppress_instrumentation
from .integrations.logging import LogfireLoggingHandler
from .integrations.structlog import LogfireProcessor as StructlogProcessor
from .version import VERSION

DEFAULT_LOGFIRE_INSTANCE: Logfire = Logfire()
span = DEFAULT_LOGFIRE_INSTANCE.span
instrument = DEFAULT_LOGFIRE_INSTANCE.instrument
force_flush = DEFAULT_LOGFIRE_INSTANCE.force_flush
log_slow_async_callbacks = DEFAULT_LOGFIRE_INSTANCE.log_slow_async_callbacks
install_auto_tracing = DEFAULT_LOGFIRE_INSTANCE.install_auto_tracing
instrument_pydantic = DEFAULT_LOGFIRE_INSTANCE.instrument_pydantic
instrument_pydantic_ai = DEFAULT_LOGFIRE_INSTANCE.instrument_pydantic_ai
instrument_asgi = DEFAULT_LOGFIRE_INSTANCE.instrument_asgi
instrument_wsgi = DEFAULT_LOGFIRE_INSTANCE.instrument_wsgi
instrument_fastapi = DEFAULT_LOGFIRE_INSTANCE.instrument_fastapi
instrument_openai = DEFAULT_LOGFIRE_INSTANCE.instrument_openai
instrument_openai_agents = DEFAULT_LOGFIRE_INSTANCE.instrument_openai_agents
instrument_anthropic = DEFAULT_LOGFIRE_INSTANCE.instrument_anthropic
instrument_tools = DEFAULT_LOGFIRE_INSTANCE.instrument_tools
instrument_asyncpg = DEFAULT_LOGFIRE_INSTANCE.instrument_asyncpg
instrument_httpx = DEFAULT_LOGFIRE_INSTANCE.instrument_httpx
instrument_celery = DEFAULT_LOGFIRE_INSTANCE.instrument_celery
instrument_requests = DEFAULT_LOGFIRE_INSTANCE.instrument_requests
instrument_psycopg = DEFAULT_LOGFIRE_INSTANCE.instrument_psycopg
instrument_django = DEFAULT_LOGFIRE_INSTANCE.instrument_django
instrument_flask = DEFAULT_LOGFIRE_INSTANCE.instrument_flask
instrument_starlette = DEFAULT_LOGFIRE_INSTANCE.instrument_starlette
instrument_aiohttp_client = DEFAULT_LOGFIRE_INSTANCE.instrument_aiohttp_client
instrument_sqlalchemy = DEFAULT_LOGFIRE_INSTANCE.instrument_sqlalchemy
instrument_sqlite3 = DEFAULT_LOGFIRE_INSTANCE.instrument_sqlite3
instrument_aws_lambda = DEFAULT_LOGFIRE_INSTANCE.instrument_aws_lambda
instrument_redis = DEFAULT_LOGFIRE_INSTANCE.instrument_redis
instrument_pymongo = DEFAULT_LOGFIRE_INSTANCE.instrument_pymongo
instrument_mysql = DEFAULT_LOGFIRE_INSTANCE.instrument_mysql
instrument_system_metrics = DEFAULT_LOGFIRE_INSTANCE.instrument_system_metrics
suppress_scopes = DEFAULT_LOGFIRE_INSTANCE.suppress_scopes
shutdown = DEFAULT_LOGFIRE_INSTANCE.shutdown
with_tags = DEFAULT_LOGFIRE_INSTANCE.with_tags
# with_trace_sample_rate = DEFAULT_LOGFIRE_INSTANCE.with_trace_sample_rate
with_settings = DEFAULT_LOGFIRE_INSTANCE.with_settings

# Logging
log = DEFAULT_LOGFIRE_INSTANCE.log
trace = DEFAULT_LOGFIRE_INSTANCE.trace
debug = DEFAULT_LOGFIRE_INSTANCE.debug
info = DEFAULT_LOGFIRE_INSTANCE.info
notice = DEFAULT_LOGFIRE_INSTANCE.notice
warn = DEFAULT_LOGFIRE_INSTANCE.warn
warning = DEFAULT_LOGFIRE_INSTANCE.warning
error = DEFAULT_LOGFIRE_INSTANCE.error
fatal = DEFAULT_LOGFIRE_INSTANCE.fatal
exception = DEFAULT_LOGFIRE_INSTANCE.exception

# Metrics
metric_counter = DEFAULT_LOGFIRE_INSTANCE.metric_counter
metric_histogram = DEFAULT_LOGFIRE_INSTANCE.metric_histogram
metric_up_down_counter = DEFAULT_LOGFIRE_INSTANCE.metric_up_down_counter
metric_gauge = DEFAULT_LOGFIRE_INSTANCE.metric_gauge
metric_counter_callback = DEFAULT_LOGFIRE_INSTANCE.metric_counter_callback
metric_gauge_callback = DEFAULT_LOGFIRE_INSTANCE.metric_gauge_callback
metric_up_down_counter_callback = DEFAULT_LOGFIRE_INSTANCE.metric_up_down_counter_callback


def save_local_logs():
    """
    Set up logging for Agno with logs saved to a date-based file in the logs directory.
    
    The logs are saved to a file named 'agno_YYYY-MM-DD_HH-MM-SS.log' in the logs directory.
    Each log line will be prefixed with a timestamp in HH:MM:SS.mmm format (time with milliseconds).
    """
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Get current date and time for filename
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_filename = f'logs/agno_{timestamp}.log'
    
    # Configure the agno logger
    agno_logger = logging.getLogger("agno")
    
    # Create file handler with the date-based filename
    file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
    
    # Create a formatter with time and milliseconds at the beginning of each line
    # %f gives microseconds (6 digits), so we use a slice to get only milliseconds (3 digits)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(message)s', datefmt='%H:%M:%S')
    file_handler.setFormatter(formatter)
    
    agno_logger.addHandler(file_handler)


def loguru_handler() -> Any:
    """Create a **Logfire** handler for Loguru.

    Returns:
        A dictionary with the handler and format for Loguru.
    """
    from .integrations import loguru

    return {'sink': loguru.LogfireHandler(), 'format': '{message}'}


__version__ = VERSION

__all__ = (
    'Logfire',
    'LogfireSpan',
    'LevelName',
    'AdvancedOptions',
    'ConsoleOptions',
    'CodeSource',
    'PydanticPlugin',
    'configure',
    'span',
    'instrument',
    'log',
    'trace',
    'debug',
    'notice',
    'info',
    'warn',
    'warning',
    'error',
    'exception',
    'fatal',
    'force_flush',
    'log_slow_async_callbacks',
    'install_auto_tracing',
    'instrument_asgi',
    'instrument_wsgi',
    'instrument_pydantic',
    'instrument_pydantic_ai',
    'instrument_fastapi',
    'instrument_openai',
    'instrument_openai_agents',
    'instrument_anthropic',
    'instrument_tools',
    'instrument_asyncpg',
    'instrument_httpx',
    'instrument_celery',
    'instrument_requests',
    'instrument_psycopg',
    'instrument_django',
    'instrument_flask',
    'instrument_starlette',
    'instrument_aiohttp_client',
    'instrument_sqlalchemy',
    'instrument_sqlite3',
    'instrument_aws_lambda',
    'instrument_redis',
    'instrument_pymongo',
    'instrument_mysql',
    'instrument_system_metrics',
    'AutoTraceModule',
    'with_tags',
    'with_settings',
    # 'with_trace_sample_rate',
    'suppress_scopes',
    'shutdown',
    'no_auto_trace',
    'ScrubMatch',
    'ScrubbingOptions',
    'VERSION',
    'add_non_user_code_prefix',
    'suppress_instrumentation',
    'StructlogProcessor',
    'LogfireLoggingHandler',
    'loguru_handler',
    'SamplingOptions',
    'MetricsOptions',
    'logfire_info',
    'save_local_logs',
)
