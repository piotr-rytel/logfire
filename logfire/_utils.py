from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, TypedDict, TypeVar, Union

from opentelemetry import trace as trace_api
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.util.instrumentation import InstrumentationScope
from opentelemetry.trace.status import Status
from opentelemetry.util import types as otel_types
from requests import RequestException, Response

T = TypeVar('T')

JsonValue = Union[int, float, str, bool, None, List['JsonValue'], Tuple['JsonValue', ...], 'JsonDict']
JsonDict = Dict[str, JsonValue]

try:
    import pydantic_core
except Exception:  # pragma: no cover

    def dump_json(obj: JsonValue) -> str:
        return json.dumps(obj, separators=(',', ':'))
else:

    def dump_json(obj: JsonValue) -> str:
        return pydantic_core.to_json(obj).decode()


def uniquify_sequence(seq: Sequence[T]) -> list[T]:
    """Remove duplicates from a sequence preserving order."""
    seen: set[T] = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def safe_repr(obj: Any) -> str:
    """Return some kind of non-empty string representation of an object, catching exceptions."""
    try:
        result = repr(obj)
    except Exception:
        result = ''

    # If repr() returns an empty string, don't use that.
    if result:  # pragma: no branch
        return result

    try:
        return f'<{type(obj).__name__} object>'
    except Exception:
        return '<unknown (repr failed)>'


Truncatable = TypeVar('Truncatable', str, bytes, 'list[Any]', 'tuple[Any, ...]')


def truncate_string(s: str, *, max_length: int, middle: str = '...') -> str:
    """Return a string at most max_length characters long, with `middle` in the middle if truncated."""
    return truncate_sequence(s, max_length=max_length, middle=middle)


def truncate_sequence(seq: Truncatable, *, max_length: int, middle: Truncatable) -> Truncatable:
    """Return a sequence at with `len()` at most `max_length`, with `middle` in the middle if truncated."""
    if len(seq) <= max_length:
        return seq
    remaining_length = max_length - len(middle)
    half = remaining_length // 2
    return seq[:half] + middle + seq[-half:]


def read_toml_file(path: Path) -> dict[str, Any]:
    """Read a TOML file and return the parsed data.

    It wraps the `tomllib.load` function from Python 3.11 or the `tomli.load` function from older versions.
    """
    if sys.version_info >= (3, 11):  # pragma: no branch
        from tomllib import load as load_toml
    else:
        from tomli import load as load_toml  # pragma: no cover

    with path.open('rb') as f:
        data = load_toml(f)

    return data


class ReadableSpanDict(TypedDict):
    """A dictionary representation of a ReadableSpan.

    ReadableSpan is immutable, so making modified versions of it is inconvenient and slow.
    Converting a ReadableSpan to a ReadableSpanDict using span_to_dict makes it easier to modify.
    See `SpanProcessorWrapper.on_end` for an example of how this is useful.
    """

    name: str
    context: trace_api.SpanContext | None
    parent: trace_api.SpanContext | None
    resource: Resource | None
    attributes: Mapping[str, otel_types.AttributeValue]
    events: Sequence[Event]
    links: Sequence[trace_api.Link]
    kind: trace_api.SpanKind
    status: Status
    start_time: int | None
    end_time: int | None
    instrumentation_scope: InstrumentationScope | None


def span_to_dict(span: ReadableSpan) -> ReadableSpanDict:
    """See ReadableSpanDict."""
    return ReadableSpanDict(
        name=span.name,
        context=span.context,
        parent=span.parent,
        resource=span.resource,
        attributes=span.attributes or {},
        events=span.events,
        links=span.links,
        kind=span.kind,
        status=span.status,
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )


class UnexpectedResponse(RequestException):
    """An unexpected response was received from the server."""

    def __init__(self, response: Response) -> None:
        super().__init__(f'Unexpected response: {response.status_code}', response=response)

    def __str__(self) -> str:
        assert self.response is not None  # silence type checker
        try:
            body_json = self.response.json()
        except ValueError:
            try:
                body = self.response.text
            except ValueError:  # pragma: no cover
                body = '[binary data]'
        else:  # pragma: no cover
            body = json.dumps(body_json, indent=2)
        request = self.response.request
        return (
            f'Unexpected response {self.response.status_code}\n\n'
            f'{request.method} {request.url} —> {self.response.status_code}\n'
            f'body: {truncate_string(body, max_length=120)}\n'
            'If the error persists, please contact us. '
            '(See https://docs.logfire.dev/help/ for contact information.)'
        )

    @classmethod
    def raise_for_status(cls, response: Response) -> None:
        """Like the requests method, but raises a more informative exception."""
        if response.status_code not in range(200, 300):
            raise cls(response)


def ensure_data_dir_exists(data_dir: Path) -> None:
    if data_dir.exists():
        if not data_dir.is_dir():  # pragma: no cover
            raise ValueError(f'Data directory {data_dir} exists but is not a directory')
        return
    data_dir.mkdir(parents=True, exist_ok=True)
    gitignore = data_dir / '.gitignore'
    gitignore.write_text('*')
