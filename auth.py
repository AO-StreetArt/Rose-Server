import logging
from functools import wraps
from typing import Iterable, Optional, Sequence
from urllib.parse import urljoin

import jwt
from flask import abort, current_app, g, request
from jwt import InvalidTokenError, PyJWKClient

logger = logging.getLogger(__name__)


class AuthError(Exception):
    """Raised when authentication fails."""


class Auth0TokenVerifier:
    """Verifies Auth0-issued access tokens."""

    def __init__(self, domain: str, audience: str, client_id: Optional[str] = None) -> None:
        issuer = _normalize_domain(domain)
        self.issuer = issuer if issuer.endswith("/") else f"{issuer}/"
        self.audience = audience
        self.client_id = client_id
        jwks_url = urljoin(self.issuer, ".well-known/jwks.json")
        self._jwk_client = PyJWKClient(jwks_url)

    def verify(self, token: str, scopes: Optional[Iterable[str]] = None) -> dict:
        scopes = set(scopes or [])
        try:
            signing_key = self._jwk_client.get_signing_key_from_jwt(token)
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                audience=self.audience,
                issuer=self.issuer,
            )
        except InvalidTokenError as exc:
            logger.debug("Auth0 token verification failed: %s", exc)
            raise AuthError("Invalid access token.") from exc

        if scopes and not _has_required_scopes(claims.get("scope"), scopes):
            raise AuthError("Insufficient scope.")

        if self.client_id:
            # Optional: ensure the token is intended for this client.
            token_audience = claims.get("azp") or claims.get("client_id")
            if token_audience and token_audience != self.client_id:
                raise AuthError("Token azp/client_id mismatch.")

        return claims


def _normalize_domain(domain: str) -> str:
    if not domain:
        return ""
    if domain.startswith("http://") or domain.startswith("https://"):
        return domain.rstrip("/")
    return f"https://{domain.strip('/')}"


def _has_required_scopes(scope_claim: Optional[str], required_scopes: Sequence[str]) -> bool:
    token_scopes = set(scope_claim.split()) if scope_claim else set()
    return set(required_scopes).issubset(token_scopes)


def _get_bearer_token() -> str:
    auth_header = request.headers.get("Authorization", "")
    parts = auth_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise AuthError("Invalid Authorization header. Expected 'Bearer <token>'.")
    return parts[1]


def requires_auth(scopes: Optional[Iterable[str]] = None):
    """Decorator that enforces Auth0 authorization on a Flask route."""

    def decorator(view_func):
        @wraps(view_func)
        def wrapped(*args, **kwargs):
            try:
                ensure_authorized(scopes)
            except AuthError as exc:
                abort(401, description=str(exc))
            return view_func(*args, **kwargs)

        return wrapped

    return decorator


def ensure_authorized(scopes: Optional[Iterable[str]] = None) -> dict:
    verifier: Auth0TokenVerifier = current_app.extensions["auth0_verifier"]
    token = _get_bearer_token()
    claims = verifier.verify(token, scopes)
    g.access_token_claims = claims
    return claims
