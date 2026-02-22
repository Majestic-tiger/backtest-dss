# -*- coding: utf-8 -*-
"""Shared UI utilities for Streamlit pages."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

# Re-export from engine so existing callers keep working
from dongpa_engine import compute_trade_metrics  # noqa: F401

# ---------------------- Constants ----------------------

NAV_LINKS = [
    ("backtest.py", "backtest"),
    ("pages/2_orderBook.py", "orderBook"),
    ("pages/3_Optuna.py", "Optuna"),
]

SETTINGS_PATH = Path("config") / "strategy.json"
LOCAL_SETTINGS_PATH = Path("config") / "personal_settings.json"
CONFIG_DIR = Path("config")
LOOKBACK_DAYS = 1000  # Extra days for weekly RSI EMA warm-up convergence

LOCAL_KEYS = {"start_date", "init_cash", "log_scale", "spread_buy_levels", "spread_buy_step", "enable_netting", "allow_fractional"}

DEFAULT_PARAMS = {
    "target": "SOXL",
    "momentum": "QQQ",
    "bench": "SOXX",
    "log_scale": True,
    "mode_switch_strategy_index": 0,
    "ma_short": 3,
    "ma_long": 7,
    "roc_period": 4,
    "btc_ticker": "BTC-USD",
    "btc_lookback_days": 1,
    "btc_threshold_pct": 0.0,
    "rsi_high_threshold": 65.0,
    "rsi_mid_high": 60.0,
    "rsi_neutral": 50.0,
    "rsi_mid_low": 40.0,
    "rsi_low_threshold": 35.0,
    "enable_netting": True,
    "allow_fractional": False,
    "cash_limited_buy": False,
    "init_cash": 10000,
    "defense_slices": 7,
    "defense_buy": 3.0,
    "defense_tp": 0.2,
    "defense_sl": 0.0,
    "defense_hold": 30,
    "offense_slices": 7,
    "offense_buy": 5.0,
    "offense_tp": 2.5,
    "offense_sl": 0.0,
    "offense_hold": 7,
}


# ---------------------- Navigation ----------------------

def render_navigation() -> None:
    st.markdown(
        """
        <style>
        [data-testid='stSidebarNav'] {display: none;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("### Pages")
    for path, label in NAV_LINKS:
        st.sidebar.page_link(path, label=label)
    st.sidebar.divider()


# ---------------------- Settings I/O ----------------------

def _read_json(path: Path) -> dict:
    """Read a JSON file and return its contents as a dict."""
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _write_json(path: Path, data: dict) -> None:
    """Write a dict to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def load_settings(config_path: Path | None = None) -> dict:
    """Load settings from config JSON file(s).

    When loading the default settings path, strategy settings and local
    (personal) settings are merged from two separate files.
    """
    path = config_path if config_path else SETTINGS_PATH
    result = _read_json(path)

    # For the default path, also merge local settings
    if path == SETTINGS_PATH:
        result.update(_read_json(LOCAL_SETTINGS_PATH))

    return result


def save_settings(payload: dict, config_path: Path | None = None) -> None:
    """Save settings to config JSON file(s).

    When saving to the default path, personal keys (start_date, init_cash,
    log_scale) are split into personal_settings.json while strategy keys go
    to strategy.json.
    """
    path = config_path if config_path else SETTINGS_PATH

    if path == SETTINGS_PATH:
        local_data = {k: v for k, v in payload.items() if k in LOCAL_KEYS}
        strategy_data = {k: v for k, v in payload.items() if k not in LOCAL_KEYS}
        _write_json(LOCAL_SETTINGS_PATH, local_data)
        _write_json(path, strategy_data)
    else:
        _write_json(path, payload)


def get_available_config_files() -> list[Path]:
    """Get all JSON config files in the config directory.

    Excludes personal_settings.json from the listing.
    """
    if not CONFIG_DIR.exists():
        return []
    excluded = {"personal_settings.json"}
    json_files = [
        p for p in CONFIG_DIR.glob("*.json")
        if p.name not in excluded
    ]
    json_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return json_files


