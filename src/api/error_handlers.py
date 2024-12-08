from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
import logging


def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """
    Handles rate limit exceptions and returns a 429 status code.

    Args:
        request: FastAPI Request object
        exc: RateLimitExceeded: Exception raised by the rate limiter.

    Returns:
        JSONResponse: A response indicating the rate limit has been exceeded.
    """
    logging.warning(f"Rate limit exceeded for request: {request.url}")
    return JSONResponse(
        status_code=429,
        content={
            "detail": "The allowed rate limit has been exceeded. Please try again later."
        },
    )


def general_exception_handler(request: Request, exc: Exception):
    """
    Handles all uncaught exceptions and returns a generic error response.

    Args:
        request: FastAPI Request object
        exc: Exception: The uncaught exception.

    Returns:
        JSONResponse: A generic error response.
    """
    logging.error(f"Unhandled exception occurred: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again later."},
    )


def add_exception_handlers(app: FastAPI):
    """
    Registers all custom exception handlers with the FastAPI app.

    Args:
        app: FastAPI application instance.
    """
    app.add_exception_handler(RateLimitExceeded, rate_limit_handler)
    app.add_exception_handler(Exception, general_exception_handler)
