"""Standardized error responses for the API."""
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError


class APIError(Exception):
    """Base API error."""
    def __init__(self, status_code: int, detail: str, error_code: str = "unknown"):
        self.status_code = status_code
        self.detail = detail
        self.error_code = error_code


class NotFoundError(APIError):
    def __init__(self, resource: str, resource_id: str):
        super().__init__(404, f"{resource} '{resource_id}' not found", "not_found")


class ConflictError(APIError):
    def __init__(self, detail: str):
        super().__init__(409, detail, "conflict")


class BadRequestError(APIError):
    def __init__(self, detail: str):
        super().__init__(400, detail, "bad_request")


async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error_code,
            "detail": exc.detail,
        },
    )


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "detail": "Request validation failed",
            "errors": exc.errors(),
        },
    )


async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "detail": "An unexpected error occurred",
        },
    )
