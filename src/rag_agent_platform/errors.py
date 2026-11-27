"""Explicit error types. The API layer maps each one to a clear HTTP response."""


class PlatformError(Exception):
    """Base class for expected, user-facing failures."""


class ConfigError(PlatformError):
    """The configuration is incomplete or contradictory."""


class IngestError(PlatformError):
    """A document could not be loaded, parsed or stored."""


class ProviderError(PlatformError):
    """The configured model provider failed or returned something unusable."""


class ToolInputError(PlatformError):
    """An agent tool rejected its input (unsafe SQL, invalid expression, ...)."""
