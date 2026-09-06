from datetime import date, datetime

DATE_FORMAT = "%Y%m%d"


def to_date(value: str | date | datetime) -> date:
    """Accept YYYYMMDD strings, datetimes or dates - always return a date."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(value, DATE_FORMAT).date()


def to_datestr(value: str | date | datetime) -> str:
    """Inverse of to_date - the YYYYMMDD form used in filenames and public APIs."""
    if isinstance(value, str):
        # Validate round-trip so a malformed string fails here, not at file-open time
        return to_date(value).strftime(DATE_FORMAT)
    return to_date(value).strftime(DATE_FORMAT)


def to_date_list(values: list[str | date]) -> list[date]:
    return [to_date(v) for v in values]


def date_range(start: str | date, end: str | date) -> list[date]:
    """Inclusive calendar-day range. Trading-day filtering is the caller's job."""
    start_d, end_d = to_date(start), to_date(end)
    if end_d < start_d:
        raise ValueError(f"end {end_d} precedes start {start_d}")
    import pandas as pd
    return [d.date() for d in pd.date_range(start_d, end_d, freq="D")]
