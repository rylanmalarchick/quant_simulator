"""
Alert system for trading notifications.

Supports email and Slack notifications.
"""

import json
import logging
import smtplib
from dataclasses import dataclass
from datetime import datetime
from email.mime.text import MIMEText
from enum import Enum
from typing import List, Optional

import requests

from swing.config import get_settings

logger = logging.getLogger(__name__)


class AlertType(Enum):
    """Types of alerts."""

    NEW_SIGNAL = "NEW_SIGNAL"
    POSITION_OPENED = "POSITION_OPENED"
    PROFIT_TARGET_HIT = "PROFIT_TARGET_HIT"
    STOP_LOSS_HIT = "STOP_LOSS_HIT"
    EXPIRY_HIT = "EXPIRY_HIT"
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    LOW_ACCOUNT_EQUITY = "LOW_ACCOUNT_EQUITY"
    DATA_FEED_ERROR = "DATA_FEED_ERROR"
    SYSTEM_ERROR = "SYSTEM_ERROR"


@dataclass
class Alert:
    """Alert message."""

    alert_type: AlertType
    title: str
    message: str
    timestamp: datetime
    symbol: Optional[str] = None
    data: Optional[dict] = None


class AlertManager:
    """
    Manages alert notifications via email and Slack.

    Per spec:
    - NEW_SIGNAL: A new high-confidence signal triggered
    - POSITION_OPENED: Trade entry confirmed
    - PROFIT_TARGET_HIT: Position hit profit target
    - STOP_LOSS_HIT: Position stopped out
    - EXPIRY_HIT: Position held to expiry, exiting
    - DAILY_LOSS_LIMIT: Hit 5% daily loss, stop trading
    - LOW_ACCOUNT_EQUITY: Account below minimum threshold
    - DATA_FEED_ERROR: Options data fetch failed
    """

    def __init__(
        self,
        email: Optional[str] = None,
        slack_webhook: Optional[str] = None,
    ):
        settings = get_settings()

        self.email = email or settings.alert_email
        self.slack_webhook = slack_webhook or settings.alert_slack_webhook

        self.alerts_history: List[Alert] = []
        self.enabled = True

    def send_alert(
        self,
        alert_type: AlertType,
        title: str,
        message: str,
        symbol: Optional[str] = None,
        data: Optional[dict] = None,
    ) -> bool:
        """Send an alert via configured channels."""
        if not self.enabled:
            return False

        alert = Alert(
            alert_type=alert_type,
            title=title,
            message=message,
            timestamp=datetime.now(),
            symbol=symbol,
            data=data,
        )

        self.alerts_history.append(alert)

        success = True

        # Send to Slack
        if self.slack_webhook:
            if not self._send_slack(alert):
                success = False

        # Send email
        if self.email:
            if not self._send_email(alert):
                success = False

        # Always log
        log_level = (
            logging.WARNING
            if alert_type
            in (
                AlertType.STOP_LOSS_HIT,
                AlertType.DAILY_LOSS_LIMIT,
                AlertType.DATA_FEED_ERROR,
                AlertType.SYSTEM_ERROR,
            )
            else logging.INFO
        )
        logger.log(log_level, f"[{alert_type.value}] {title}: {message}")

        return success

    def _send_slack(self, alert: Alert) -> bool:
        """Send alert to Slack webhook."""
        if not self.slack_webhook:
            return False

        # Emoji based on alert type
        emoji_map = {
            AlertType.NEW_SIGNAL: ":chart_with_upwards_trend:",
            AlertType.POSITION_OPENED: ":white_check_mark:",
            AlertType.PROFIT_TARGET_HIT: ":moneybag:",
            AlertType.STOP_LOSS_HIT: ":red_circle:",
            AlertType.EXPIRY_HIT: ":hourglass:",
            AlertType.DAILY_LOSS_LIMIT: ":rotating_light:",
            AlertType.LOW_ACCOUNT_EQUITY: ":warning:",
            AlertType.DATA_FEED_ERROR: ":x:",
            AlertType.SYSTEM_ERROR: ":fire:",
        }
        emoji = emoji_map.get(alert.alert_type, ":bell:")

        # Build Slack message
        payload = {
            "text": f"{emoji} *{alert.title}*",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"{emoji} {alert.title}",
                    },
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": alert.message,
                    },
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Type:* {alert.alert_type.value} | *Time:* {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                        }
                    ],
                },
            ],
        }

        if alert.symbol:
            payload["blocks"].insert(
                2,
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Symbol:* {alert.symbol}"}
                    ],
                },
            )

        try:
            response = requests.post(
                self.slack_webhook,
                json=payload,
                timeout=10,
            )
            response.raise_for_status()
            return True

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False

    def _send_email(self, alert: Alert) -> bool:
        """Send alert via email (basic implementation)."""
        if not self.email:
            return False

        # NOTE: This is a basic implementation.
        # In production, you'd use a service like SendGrid or AWS SES.
        try:
            msg = MIMEText(
                f"""
Alert Type: {alert.alert_type.value}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Symbol: {alert.symbol or 'N/A'}

{alert.message}

Data: {json.dumps(alert.data, indent=2) if alert.data else 'N/A'}
"""
            )
            msg["Subject"] = f"[Swing Trading] {alert.title}"
            msg["From"] = "swing-trading@localhost"
            msg["To"] = self.email

            # This would need an SMTP server configured
            # For now, just log that we would send
            logger.info(f"Would send email to {self.email}: {alert.title}")

            return True

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False

    # Convenience methods for common alerts

    def signal_alert(
        self,
        symbol: str,
        direction: str,
        confidence: float,
    ) -> bool:
        """Alert for new trading signal."""
        return self.send_alert(
            AlertType.NEW_SIGNAL,
            f"New {direction} Signal: {symbol}",
            f"Signal generated with {confidence:.1%} confidence",
            symbol=symbol,
            data={"direction": direction, "confidence": confidence},
        )

    def position_opened_alert(
        self,
        symbol: str,
        direction: str,
        shares: int,
        entry_price: float,
    ) -> bool:
        """Alert for position opened."""
        return self.send_alert(
            AlertType.POSITION_OPENED,
            f"Position Opened: {symbol}",
            f"Opened {direction} position: {shares} shares @ ${entry_price:.2f}",
            symbol=symbol,
            data={
                "direction": direction,
                "shares": shares,
                "entry_price": entry_price,
            },
        )

    def stop_loss_alert(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
    ) -> bool:
        """Alert for stop loss hit."""
        return self.send_alert(
            AlertType.STOP_LOSS_HIT,
            f"Stop Loss Hit: {symbol}",
            f"Exited @ ${exit_price:.2f} (entry: ${entry_price:.2f}). "
            f"P&L: ${pnl:.2f}",
            symbol=symbol,
            data={
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
            },
        )

    def profit_target_alert(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
    ) -> bool:
        """Alert for profit target hit."""
        return self.send_alert(
            AlertType.PROFIT_TARGET_HIT,
            f"Profit Target Hit: {symbol}",
            f"Exited @ ${exit_price:.2f} (entry: ${entry_price:.2f}). "
            f"P&L: ${pnl:.2f}",
            symbol=symbol,
            data={
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
            },
        )

    def daily_loss_limit_alert(self, daily_pnl: float, limit: float) -> bool:
        """Alert for daily loss limit hit."""
        return self.send_alert(
            AlertType.DAILY_LOSS_LIMIT,
            "Daily Loss Limit Reached",
            f"Daily P&L: ${daily_pnl:.2f} exceeds limit of ${limit:.2f}. "
            "Trading halted for today.",
            data={"daily_pnl": daily_pnl, "limit": limit},
        )

    def data_feed_error_alert(self, error: str) -> bool:
        """Alert for data feed errors."""
        return self.send_alert(
            AlertType.DATA_FEED_ERROR,
            "Data Feed Error",
            f"Failed to fetch options data: {error}",
            data={"error": error},
        )
