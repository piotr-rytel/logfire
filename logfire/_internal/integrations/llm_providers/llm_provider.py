from __future__ import annotations

from collections.abc import Iterable
from contextlib import ExitStack, contextmanager, nullcontext
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, ContextManager, Iterator, Optional, cast

from ...constants import ONE_SECOND_IN_NANOSECONDS
from ...utils import is_instrumentation_suppressed, suppress_instrumentation
from ...main import LogfireSpan

if TYPE_CHECKING:
    from ...main import Logfire
    from .types import EndpointConfig, StreamState


__all__ = ('instrument_llm_provider',)


def instrument_llm_provider(
    logfire: Logfire,
    client: Any,
    suppress_otel: bool,
    scope_suffix: str,
    get_endpoint_config_fn: Callable[[Any], EndpointConfig],
    on_response_fn: Callable[[Any, LogfireSpan], Any],
    is_async_client_fn: Callable[[type[Any]], bool],
) -> ContextManager[None]:
    """Instruments the provided `client` (or clients) with `logfire`.

    The `client` argument can be:
    - a single client instance, e.g. an instance of `openai.OpenAI`,
    - a class of a client, or
    - an iterable of clients/classes.

    Returns:
        A context manager that will revert the instrumentation when exited.
            Use of this context manager is optional.
    """
    if isinstance(client, Iterable):
        # Eagerly instrument each client, but only open the returned context managers
        # in another context manager which the user needs to open if they want.
        # Otherwise the garbage collector will close them and uninstrument.
        context_managers = [
            instrument_llm_provider(
                logfire,
                c,
                suppress_otel,
                scope_suffix,
                get_endpoint_config_fn,
                on_response_fn,
                is_async_client_fn,
            )
            for c in cast('Iterable[Any]', client)
        ]

        @contextmanager
        def uninstrument_context():
            with ExitStack() as exit_stack:
                for context_manager in context_managers:
                    exit_stack.enter_context(context_manager)
                yield

        return uninstrument_context()

    if getattr(client, '_is_instrumented_by_logfire', False):
        # Do nothing if already instrumented.
        return nullcontext()

    logfire_llm = logfire.with_settings(custom_scope_suffix=scope_suffix.lower(), tags=['LLM'])

    client._is_instrumented_by_logfire = True
    client._original_request_method = original_request_method = client._request

    is_async = is_async_client_fn(client if isinstance(client, type) else type(client))

    # Extract LLM provider name from scope_suffix
    llm_name = scope_suffix.strip()

    def _instrumentation_setup(**kwargs: Any) -> Any:
        if is_instrumentation_suppressed():
            return None, None, kwargs

        message_template, span_data, stream_state_cls = get_endpoint_config_fn(kwargs['options'])
        if not message_template:
            return None, None, kwargs

        span_data['async'] = is_async
        model_name = span_data.get('request_data', {}).get('model', 'unknown')
        
        # Different parent message templates for streaming and non-streaming
        stream = kwargs['stream']
        if stream:
            parent_message_template = f"LLM {llm_name} Stream Call: {model_name}"
        else:
            parent_message_template = f"LLM {llm_name} Call: {model_name}"
        
        span_data['parent_message_template'] = parent_message_template

        if stream and stream_state_cls:
            stream_cls = kwargs['stream_cls']
            assert stream_cls is not None, 'Expected `stream_cls` when streaming'

            if is_async:

                class LogfireInstrumentedAsyncStream(stream_cls):
                    async def __stream__(self) -> AsyncIterator[Any]:
                        # Create only the parent span for streaming
                        with logfire_llm.span(parent_message_template, **span_data) as parent_span:
                            # Pass the parent span to record_streaming
                            with record_streaming(logfire_llm, span_data, stream_state_cls, parent_span) as record_chunk:
                                async for chunk in super().__stream__():  # type: ignore
                                    record_chunk(chunk)
                                    yield chunk

                kwargs['stream_cls'] = LogfireInstrumentedAsyncStream
            else:

                class LogfireInstrumentedStream(stream_cls):
                    def __stream__(self) -> Iterator[Any]:
                        # Create only the parent span for streaming
                        with logfire_llm.span(parent_message_template, **span_data) as parent_span:
                            # Pass the parent span to record_streaming
                            with record_streaming(logfire_llm, span_data, stream_state_cls, parent_span) as record_chunk:
                                for chunk in super().__stream__():  # type: ignore
                                    record_chunk(chunk)
                                    yield chunk

                kwargs['stream_cls'] = LogfireInstrumentedStream

        return message_template, span_data, kwargs

    # In these methods, `*args` is only expected to be `(self,)`
    # in the case where we instrument classes rather than client instances.

    def instrumented_llm_request_sync(*args: Any, **kwargs: Any) -> Any:
        message_template, span_data, kwargs = _instrumentation_setup(**kwargs)
        if message_template is None:
            return original_request_method(*args, **kwargs)
        
        stream = kwargs['stream']
        model_name = span_data.get('request_data', {}).get('model', 'unknown')
        parent_message_template = span_data.get('parent_message_template')
        
        if stream:
            # For streaming requests, the parent span is created in the stream class
            # Don't create a span here to avoid duplicates
            with maybe_suppress_instrumentation(suppress_otel):
                return original_request_method(*args, **kwargs)
        else:
            # Create only the parent span for non-streaming requests
            with logfire_llm.span(parent_message_template, **span_data) as parent_span:
                with maybe_suppress_instrumentation(suppress_otel):
                    response = original_request_method(*args, **kwargs)
                    # Extract model from response before passing to on_response_fn
                    if hasattr(response, 'model'):
                        parent_span.set_attribute('response_model', response.model)
                    elif isinstance(response, dict) and 'model' in response:
                        parent_span.set_attribute('response_model', response['model'])
                    return on_response_fn(response, parent_span)

    async def instrumented_llm_request_async(*args: Any, **kwargs: Any) -> Any:
        message_template, span_data, kwargs = _instrumentation_setup(**kwargs)
        if message_template is None:
            return await original_request_method(*args, **kwargs)
        
        stream = kwargs['stream']
        model_name = span_data.get('request_data', {}).get('model', 'unknown')
        parent_message_template = span_data.get('parent_message_template')
        
        if stream:
            # For streaming requests, the parent span is created in the stream class
            # Don't create a span here to avoid duplicates
            with maybe_suppress_instrumentation(suppress_otel):
                return await original_request_method(*args, **kwargs)
        else:
            # Create only the parent span for non-streaming requests
            with logfire_llm.span(parent_message_template, **span_data) as parent_span:
                with maybe_suppress_instrumentation(suppress_otel):
                    response = await original_request_method(*args, **kwargs)
                    # Extract model from response before passing to on_response_fn
                    if hasattr(response, 'model'):
                        parent_span.set_attribute('response_model', response.model)
                    elif isinstance(response, dict) and 'model' in response:
                        parent_span.set_attribute('response_model', response['model'])
                    return on_response_fn(response, parent_span)

    if is_async:
        client._request = instrumented_llm_request_async
    else:
        client._request = instrumented_llm_request_sync

    @contextmanager
    def uninstrument_context():
        """Context manager to remove instrumentation from the LLM client.

        The user isn't required (or even expected) to use this context manager,
        which is why the instrumenting has already happened before.
        It exists mostly for tests and just in case users want it.
        """
        try:
            yield
        finally:
            client._request = client._original_request_method  # type: ignore
            del client._original_request_method
            client._is_instrumented_by_logfire = False

    return uninstrument_context()


@contextmanager
def maybe_suppress_instrumentation(suppress: bool) -> Iterator[None]:
    if suppress:
        with suppress_instrumentation():
            yield
    else:
        yield


@contextmanager
def record_streaming(
    logire_llm: Logfire,
    span_data: dict[str, Any],
    stream_state_cls: type[StreamState],
    parent_span: Optional[LogfireSpan] = None,
):
    stream_state = stream_state_cls()

    def record_chunk(chunk: Any) -> None:
        if chunk:
            # Extract model from chunk if available
            if hasattr(chunk, 'model') and parent_span:
                parent_span.set_attribute('response_model', chunk.model)
            elif isinstance(chunk, dict) and 'model' in chunk and parent_span:
                parent_span.set_attribute('response_model', chunk['model'])
            stream_state.record_chunk(chunk)

    timer = logire_llm._config.advanced.ns_timestamp_generator  # type: ignore
    start = timer()
    try:
        yield record_chunk
    finally:
        duration = (timer() - start) / ONE_SECOND_IN_NANOSECONDS
        model_name = span_data.get('request_data', {}).get('model', 'unknown')
        response_data = stream_state.get_response_data()
        
        # If parent_span is provided, add all data to it
        if parent_span:
            parent_span.set_attribute('response_data', response_data)
            parent_span.set_attribute('streaming_duration', duration)
            # Add a log message to the parent span with the correct format
            parent_span.add_event(
                f"streaming response from '{model_name}' took {duration:.2f}s",
                {
                    'duration': duration,
                }
            )
        else:
            # Fallback to the original behavior
            logire_llm.info(
                'streaming response from {request_data[model]!r} took {duration:.2f}s',
                **span_data,
                duration=duration,
                response_data=response_data,
            )