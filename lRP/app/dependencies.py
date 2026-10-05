from fastapi import Depends, HTTPException, Request

from app.auth import public_account


async def require_user(request: Request) -> dict:
    username = request.session.get("username")
    account = public_account(username) if isinstance(username, str) else None
    if account is None:
        raise HTTPException(status_code=303, headers={"Location": "/login"})
    request.state.user = account
    return account


def require_roles(*roles: str):
    async def role_dependency(
        request: Request,
        user: dict = Depends(require_user),
    ) -> dict:
        if user["role"] not in roles:
            raise HTTPException(status_code=403)
        return user

    return role_dependency