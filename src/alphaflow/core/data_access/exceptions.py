class PITViolationError(Exception):
    """Raised when a research-mode read would return data stamped after as_of_date."""


class DataNotFoundError(Exception):
    """Raised when a source returns nothing for the requested RICs / date range."""


class DataSourceError(Exception):
    """Raised when an underlying source (desktool, S3) fails."""
