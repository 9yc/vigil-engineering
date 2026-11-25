"""Independently usable support modules extracted from the project's host code."""

from .adb_session import AdbError, AdbSession, CommandResult, RetryPolicy, TcpStreamForwarder, validate_serial
from .binary_codec import FrameReader, FrameWriter
from .json_output import parse_output, validate_calls
from .process_link import LineProcess

__all__ = [
    "AdbError",
    "AdbSession",
    "CommandResult",
    "FrameReader",
    "FrameWriter",
    "LineProcess",
    "RetryPolicy",
    "TcpStreamForwarder",
    "parse_output",
    "validate_calls",
    "validate_serial",
]
