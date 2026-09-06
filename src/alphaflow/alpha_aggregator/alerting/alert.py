from alphaflow.core.integrations.logging_setup import setup_logger

log = setup_logger(__name__)


def send_critical_alert(subject: str, message: str) -> None:
    """Logs CRITICAL, then hands off to the shared alert function (email + beep).

    That function lives in another internal package and is not built here - this is the
    integration point. The CRITICAL log always fires, so an alert path that is missing or
    itself broken can never swallow the incident.
    """
    log.critical(f"{subject}: {message}")
    try:
        from apcr_desktool.alerting import send_alert
    except ImportError:
        log.warning("no external alert function available - the CRITICAL log above is the only notification")
        return
    try:
        send_alert(subject=subject, message=message)
    except Exception as exc:
        log.error(f"external alert function failed: {exc}")
