"""
exception_hierarchy.py
======================
Defines a production-grade exception hierarchy and a global exception handler:
  - AppError           : base class (all application exceptions inherit from this)
  - ValidationError    : invalid input data
  - DatabaseError      : persistence layer failures
  - RetryableError     : transient failures that should be retried
  - ConfigurationError : bad or missing configuration
  - ResourceError      : external resource unavailable

Also installs a global sys.excepthook to ensure unhandled exceptions are
structured-logged before the process exits.
"""

from __future__ import annotations

import logging
import sys
import traceback
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Exception Hierarchy
# ---------------------------------------------------------------------------
class AppError(Exception):
    """Base class for all application-level exceptions.

    Every AppError carries:
      - message   : human-readable description
      - code      : machine-readable short code for monitoring dashboards
      - context   : arbitrary key-value dict for structured logging
    """

    default_code: str = "APP_ERROR"
    http_status: int = 500

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.default_code
        self.context: dict[str, Any] = context or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_type": type(self).__name__,
            "code": self.code,
            "message": self.message,
            "context": self.context,
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}(code={self.code!r}, message={self.message!r})"


class ValidationError(AppError):
    """Raised when input data fails schema or business-rule validation."""

    default_code = "VALIDATION_ERROR"
    http_status = 422

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Any = None,
        **kwargs: Any,
    ) -> None:
        ctx = kwargs.pop("context", {})
        if field is not None:
            ctx["field"] = field
        if value is not None:
            ctx["rejected_value"] = repr(value)
        super().__init__(message, context=ctx, **kwargs)
        self.field = field
        self.value = value


class DatabaseError(AppError):
    """Raised for persistence layer failures (query errors, connection issues)."""

    default_code = "DATABASE_ERROR"
    http_status = 503

    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        ctx = kwargs.pop("context", {})
        if query:
            ctx["query_fragment"] = query[:200]  # truncate long queries
        super().__init__(message, context=ctx, **kwargs)
        self.query = query


class RetryableError(AppError):
    """Transient failure that the caller should retry after a delay.

    Carries a suggested ``retry_after_seconds`` hint.
    """

    default_code = "RETRYABLE_ERROR"
    http_status = 503

    def __init__(
        self,
        message: str,
        retry_after_seconds: float = 1.0,
        **kwargs: Any,
    ) -> None:
        ctx = kwargs.pop("context", {})
        ctx["retry_after_seconds"] = retry_after_seconds
        super().__init__(message, context=ctx, **kwargs)
        self.retry_after_seconds = retry_after_seconds


class ConfigurationError(AppError):
    """Raised when required configuration is missing or invalid."""

    default_code = "CONFIGURATION_ERROR"
    http_status = 500

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        ctx = kwargs.pop("context", {})
        if config_key:
            ctx["config_key"] = config_key
        super().__init__(message, context=ctx, **kwargs)
        self.config_key = config_key


class ResourceError(AppError):
    """External resource (cache, queue, object store) is unavailable."""

    default_code = "RESOURCE_ERROR"
    http_status = 503

    def __init__(
        self,
        message: str,
        resource_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        ctx = kwargs.pop("context", {})
        if resource_name:
            ctx["resource"] = resource_name
        super().__init__(message, context=ctx, **kwargs)
        self.resource_name = resource_name


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------
def install_global_exception_handler(logger: Optional[logging.Logger] = None) -> None:
    """Replace sys.excepthook with a structured-logging version.

    Unhandled exceptions will be logged as ERROR with full traceback before
    the default Python crash behaviour occurs.
    """
    _log = logger or logging.getLogger("global_exception_handler")

    def _handler(
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_tb: Any,
    ) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            # Let Ctrl-C work normally
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return

        tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))

        if isinstance(exc_value, AppError):
            _log.critical(
                "Unhandled AppError: %s  | structured=%s\n%s",
                exc_value,
                exc_value.to_dict(),
                tb_str,
            )
        else:
            _log.critical(
                "Unhandled exception [%s]: %s\n%s",
                exc_type.__name__,
                exc_value,
                tb_str,
            )

    sys.excepthook = _handler
    _log.debug("Global exception handler installed")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def demo_exception_hierarchy(logger: logging.Logger) -> None:
    logger.info("=== Exception Hierarchy demo ===")

    # ValidationError
    try:
        raise ValidationError(
            "Email address is not valid",
            field="email",
            value="not-an-email",
        )
    except ValidationError as exc:
        logger.error("Caught ValidationError: %s", exc.to_dict())

    # DatabaseError
    try:
        raise DatabaseError(
            "Connection pool exhausted",
            query="SELECT * FROM large_table WHERE ...",
            code="DB_POOL_EXHAUSTED",
        )
    except DatabaseError as exc:
        logger.error("Caught DatabaseError: %s", exc.to_dict())

    # RetryableError
    try:
        raise RetryableError("Upstream service returned 429", retry_after_seconds=5.0)
    except RetryableError as exc:
        logger.warning(
            "RetryableError: retry in %.1f s  %s",
            exc.retry_after_seconds,
            exc.to_dict(),
        )

    # ConfigurationError
    try:
        raise ConfigurationError("Missing required key", config_key="service.api_key")
    except ConfigurationError as exc:
        logger.error("ConfigurationError: %s", exc.to_dict())

    # ResourceError
    try:
        raise ResourceError("Redis unreachable", resource_name="session-cache")
    except ResourceError as exc:
        logger.error("ResourceError: %s", exc.to_dict())

    # Catch-all via base class
    errors = [
        ValidationError("bad field"),
        DatabaseError("query failed"),
        RetryableError("timeout"),
    ]
    for err in errors:
        if isinstance(err, AppError):
            logger.info("Caught as AppError: code=%s", err.code)


def main() -> None:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("exception_hierarchy")
    install_global_exception_handler(logger)
    demo_exception_hierarchy(logger)
    logger.info("exception_hierarchy demo complete.")


if __name__ == "__main__":
    main()
