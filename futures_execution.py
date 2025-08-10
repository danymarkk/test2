"""USDⓈ-M Futures Execution Scaffold (Binance USDM via ccxt)

Provides a minimal futures execution engine with:
- Exchange setup (testnet via sandbox mode)
- Leverage and margin-mode configuration per symbol
- Risk-based contract sizing with market precision/limits
- Entry orders (market/limit)
- Bracket exits as separate reduceOnly orders (TP/SL)
- Dry-run mode that logs instead of placing orders

This is a scaffold. It is designed to be safe by default (dry_run=True).
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import ccxt

from params import get_config, get_logger


class ExecutionEngineFutures:
    def __init__(
        self,
        api_key: Optional[str] = None,
        secret: Optional[str] = None,
        testnet: Optional[bool] = None,
        dry_run: bool = True,
        exchange_id: str = "binanceusdm",
    ) -> None:
        cfg = get_config()
        self.logger = get_logger("futures_execution")
        self.dry_run = bool(dry_run)

        # Prefer explicit arg, else config
        use_testnet = (
            cfg.get("testnet", False) if testnet is None else bool(testnet)
        )

        exchange_cls = getattr(ccxt, exchange_id)
        self.exchange = exchange_cls(
            {
                "apiKey": api_key or cfg.get("api_key", None),
                "secret": secret or cfg.get("secret", None),
                "enableRateLimit": True,
                # defaultType on binanceusdm is futures by default; keep explicit
                "options": {"defaultType": "future"},
            }
        )
        try:
            self.exchange.set_sandbox_mode(use_testnet)
        except Exception:
            # Older ccxt: ignore if not supported
            pass

        try:
            self.exchange.load_markets()
        except Exception as e:
            self.logger.warning("load_markets failed: %s", e)

        # Defaults
        self.default_leverage = int(cfg.get("leverage", 3))
        self.margin_mode = str(cfg.get("margin_mode", "isolated")).lower()
        self.post_only = bool(cfg.get("post_only", False))
        self.reduce_only = bool(cfg.get("reduce_only", True))

    # --- Utilities ---
    def _market(self, symbol: str) -> Dict[str, Any]:
        return self.exchange.market(symbol)

    def round_amount(self, symbol: str, amount: float) -> float:
        # Prefer ccxt precision helpers
        try:
            amt = float(amount)
            amt = float(self.exchange.amount_to_precision(symbol, amt))
        except Exception:
            # Fallback to market precision/limits if helpers not available
            market = self._market(symbol)
            precision = market.get("precision", {}).get("amount")
            if precision is not None:
                step = 10 ** (-precision)
                amt = math.floor(amount / step) * step
            else:
                amt = float(amount)
            minimum = market.get("limits", {}).get("amount", {}).get("min") or 0.0
            if minimum and amt < minimum:
                amt = float(minimum)
        # Note: minNotional checks require price; validation occurs on order placement
        return float(amt)

    def risk_to_qty(
        self,
        symbol: str,
        balance: float,
        risk_pct: float,
        entry_price: float,
        stop_price: float,
    ) -> float:
        risk_amount = max(0.0, balance) * max(0.0, risk_pct)
        price_risk = abs(entry_price - stop_price)
        if price_risk <= 0:
            return 0.0
        qty = risk_amount / price_risk
        return self.round_amount(symbol, qty)

    # --- Exchange configuration ---
    def set_leverage(self, symbol: str, leverage: Optional[int] = None) -> None:
        lev = int(leverage or self.default_leverage)
        if self.dry_run:
            self.logger.info("[DRY] set_leverage %s -> %dx", symbol, lev)
            return
        try:
            self.exchange.set_leverage(lev, symbol)
            self.logger.info("Leverage set %s -> %dx", symbol, lev)
        except Exception as e:
            self.logger.error("set_leverage failed for %s: %s", symbol, e)

    def set_margin_mode(self, symbol: str, mode: Optional[str] = None) -> None:
        m = (mode or self.margin_mode).upper()
        if self.dry_run:
            self.logger.info("[DRY] set_margin_mode %s -> %s", symbol, m)
            return
        try:
            # ccxt unified: set_margin_mode(mode, symbol)
            self.exchange.set_margin_mode(m, symbol)
            self.logger.info("Margin mode set %s -> %s", symbol, m)
        except Exception as e:
            self.logger.error("set_margin_mode failed for %s: %s", symbol, e)

    # --- Orders ---
    def place_entry(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "market",
        price: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        p = dict(params or {})
        if order_type == "limit":
            p["postOnly"] = p.get("postOnly", self.post_only)
        # Validate and round
        try:
            qty = float(self.exchange.amount_to_precision(symbol, qty))
        except Exception:
            qty = self.round_amount(symbol, qty)
        try:
            if price is not None:
                price = float(self.exchange.price_to_precision(symbol, price))
        except Exception:
            pass
        if self.dry_run:
            self.logger.info(
                (
                    "[DRY] entry %s %s qty=%s type=%s price=%s params=%s"
                ),
                symbol,
                side,
                qty,
                order_type,
                price,
                p,
            )
            return {"dry_run": True, "symbol": symbol, "side": side, "qty": qty}
        try:
            if order_type == "market":
                order = self.exchange.create_order(
                    symbol, "market", side, qty, None, p
                )
            else:
                order = self.exchange.create_order(
                    symbol, "limit", side, qty, price, p
                )
            self.logger.info("Entry order placed: %s", order.get("id"))
            return order
        except Exception as e:
            self.logger.error("Entry order failed: %s", e)
            return {"error": str(e)}

    def place_reduce_only(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        tag: str = "",
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"reduceOnly": True}
        if stop_price is not None:
            params["stopPrice"] = stop_price
        # Validate and round qty/price
        try:
            qty = float(self.exchange.amount_to_precision(symbol, qty))
        except Exception:
            qty = self.round_amount(symbol, qty)
        try:
            if price is not None:
                price = float(self.exchange.price_to_precision(symbol, price))
        except Exception:
            pass
        try:
            if stop_price is not None:
                stop_price = float(self.exchange.price_to_precision(symbol, stop_price))
        except Exception:
            pass
        if self.dry_run:
            self.logger.info(
                "[DRY] reduceOnly %s %s qty=%s type=%s price=%s stop=%s",
                symbol,
                side,
                qty,
                order_type,
                price,
                stop_price,
            )
            return {"dry_run": True, "symbol": symbol, "side": side, "qty": qty, "tag": tag}
        try:
            order = self.exchange.create_order(
                symbol, order_type, side, qty, price, params
            )
            self.logger.info("%s reduceOnly placed: %s", tag, order.get("id"))
            return order
        except Exception as e:
            self.logger.error("reduceOnly order failed: %s", e)
            return {"error": str(e)}

    def close_position(self, symbol: str, side: str, qty: float) -> Dict[str, Any]:
        # side should be opposite of position
        return self.place_reduce_only(symbol, side, qty, "market", None, None, tag="CLOSE")


def discover_futures_symbols_usdm(testnet: bool = True) -> list[str]:
    """Discover linear swap futures with USDT/USDC quote on Binance USDM."""
    log = get_logger("futures_execution")
    ex = ccxt.binanceusdm({"enableRateLimit": True, "options": {"defaultType": "future"}})
    try:
        ex.set_sandbox_mode(bool(testnet))
    except Exception:
        pass
    try:
        markets = ex.load_markets()
    except Exception as e:
        log.error("load_markets failed: %s", e)
        return []
    symbols: list[str] = []
    for m in markets.values():
        try:
            if m.get("swap") and m.get("linear") and m.get("quote") in {"USDT", "USDC"}:
                symbols.append(m["symbol"])
        except Exception:
            continue
    symbols = sorted(set(symbols))
    log.info("Discovered %d futures symbols", len(symbols))
    return symbols


