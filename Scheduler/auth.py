"""ZITADEL access-token authentication."""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass

import httpx
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


bearer = HTTPBearer(auto_error=False)


@dataclass
class CachedIdentity:
    expires_at: float
    claims: dict


class ZitadelAuthenticator:
    def __init__(self) -> None:
        self.issuer = os.getenv("ZITADEL_ISSUER", "").rstrip("/")
        self.client_id = os.getenv("ZITADEL_INTROSPECTION_CLIENT_ID", "")
        self.client_secret = os.getenv("ZITADEL_INTROSPECTION_CLIENT_SECRET", "")
        self.audience = os.getenv("ZITADEL_AUDIENCE", "")
        self.cache: dict[str, CachedIdentity] = {}

    @property
    def configured(self) -> bool:
        return bool(self.issuer and self.client_id and self.client_secret)

    @staticmethod
    def _audiences(claims: dict) -> set[str]:
        audience = claims.get("aud", [])
        if isinstance(audience, str):
            return {audience}
        return set(audience or [])

    async def identity(self, token: str) -> dict:
        if not self.configured:
            raise HTTPException(503, "ZITADEL authentication is not configured on the server.")
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        cached = self.cache.get(token_hash)
        now = time.time()
        if cached and cached.expires_at > now:
            return cached.claims

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                introspection_response = await client.post(
                    f"{self.issuer}/oauth/v2/introspect",
                    data={"token": token, "token_type_hint": "access_token"},
                    auth=(self.client_id, self.client_secret),
                    headers={"Accept": "application/json"},
                )
                introspection_response.raise_for_status()
                claims = introspection_response.json()
                if not claims.get("active") or not claims.get("sub"):
                    raise HTTPException(401, "The access token is inactive or invalid.")
                if self.audience and self.audience not in self._audiences(claims):
                    raise HTTPException(403, "The access token was not issued for PeopleWelcome.")
                userinfo_response = await client.get(
                    f"{self.issuer}/oidc/v1/userinfo",
                    headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
                )
                if userinfo_response.is_success:
                    claims.update(userinfo_response.json())
        except HTTPException:
            raise
        except (httpx.HTTPError, ValueError) as error:
            raise HTTPException(503, "ZITADEL could not validate the access token.") from error

        token_expiry = float(claims.get("exp", now + 60))
        self.cache[token_hash] = CachedIdentity(min(now + 60, token_expiry), claims)
        if len(self.cache) > 1000:
            self.cache = {key: value for key, value in self.cache.items() if value.expires_at > now}
        return claims


authenticator = ZitadelAuthenticator()


async def authenticated_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials,
) -> dict:
    claims = await authenticator.identity(credentials.credentials)
    username = claims.get("preferred_username") or claims.get("name") or claims.get("email") or claims["sub"]
    return request.app.state.store.upsert_user(claims["sub"], username, claims.get("email"))


async def require_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> dict:
    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(401, "Sign in with ZITADEL to continue.")
    return await authenticated_user(request, credentials)


async def optional_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> dict | None:
    if not credentials:
        return None
    if credentials.scheme.lower() != "bearer":
        raise HTTPException(401, "Invalid authorization scheme.")
    return await authenticated_user(request, credentials)
