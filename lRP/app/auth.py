from pwdlib import PasswordHash

from app.config import ACCOUNTS

password_hasher = PasswordHash.recommended()
_accounts_by_name = {
    account["username"]: {
        "username": account["username"],
        "role": account["role"],
        "password_hash": password_hasher.hash(account["password"]),
    }
    for account in ACCOUNTS.values()
}


def authenticate(username: str, password: str) -> dict | None:
    account = _accounts_by_name.get(username)
    if account is None or not password_hasher.verify(password, account["password_hash"]):
        return None
    return {"username": account["username"], "role": account["role"]}


def public_account(username: str) -> dict | None:
    account = _accounts_by_name.get(username)
    if account is None:
        return None
    return {"username": account["username"], "role": account["role"]}