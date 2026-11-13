"""Independently usable support modules extracted from the project's host code."""

from .binary_codec import FrameReader, FrameWriter
from .json_output import parse_output, validate_calls
from .process_link import LineProcess

__all__ = ["FrameReader", "FrameWriter", "LineProcess", "parse_output", "validate_calls"]
