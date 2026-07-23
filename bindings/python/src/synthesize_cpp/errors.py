"""Public Python Adapter exceptions."""


class SynthesizeError(RuntimeError):
    """Base error raised by the synthesize.cpp Python Adapter."""


class ProviderError(SynthesizeError):
    """The installed Native Provider set violates the Adapter contract."""
