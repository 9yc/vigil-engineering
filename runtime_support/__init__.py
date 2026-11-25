"""Independently usable support modules extracted from the project's host code."""

from .adb_session import AdbError, AdbSession, CommandResult, RetryPolicy, TcpStreamForwarder, validate_serial
from .binary_codec import FrameReader, FrameWriter
from .json_output import parse_output, validate_calls
from .power_telemetry import (
    BatterySnapshot,
    integrate_coulomb_energy,
    integrate_vi_trapezoidal,
    parse_dumpsys_battery,
    parse_proc_meminfo,
    parse_proc_uptime,
    subtract_standby_baseline,
)
from .process_link import LineProcess
from .thermal_duty import (
    BatteryLowError,
    DutyCycleController,
    ProbeReading,
    ThermalGateConfig,
    ThermalGateError,
    ThermalTimeoutError,
)

__all__ = [
    "AdbError",
    "AdbSession",
    "BatteryLowError",
    "BatterySnapshot",
    "CommandResult",
    "DutyCycleController",
    "FrameReader",
    "FrameWriter",
    "LineProcess",
    "ProbeReading",
    "RetryPolicy",
    "TcpStreamForwarder",
    "ThermalGateConfig",
    "ThermalGateError",
    "ThermalTimeoutError",
    "integrate_coulomb_energy",
    "integrate_vi_trapezoidal",
    "parse_dumpsys_battery",
    "parse_output",
    "parse_proc_meminfo",
    "parse_proc_uptime",
    "subtract_standby_baseline",
    "validate_calls",
    "validate_serial",
]
