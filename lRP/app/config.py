import os

APP_ENV = os.getenv("APP_ENV", "development").strip().lower()
if APP_ENV == "production":
    session_secret = os.getenv("SESSION_SECRET_KEY", "")
    if len(session_secret) < 32:
        raise RuntimeError(
            "SESSION_SECRET_KEY must contain at least 32 characters in production"
        )
if APP_ENV == "production":
    required_account_settings = (
        "ADMIN_USERNAME",
        "ADMIN_PASSWORD",
        "ANALYST_USERNAME",
        "ANALYST_PASSWORD",
        "VIEWER_USERNAME",
        "VIEWER_PASSWORD",
    )
    missing_settings = [name for name in required_account_settings if not os.getenv(name)]
    if missing_settings:
        raise RuntimeError(
            "Set production account environment variables: " + ", ".join(missing_settings)
        )
    development_passwords = {
        "ADMIN_PASSWORD": "AdminDemo123!",
        "ANALYST_PASSWORD": "AnalystDemo123!",
        "VIEWER_PASSWORD": "ViewerDemo123!",
    }
    unchanged_passwords = [
        name
        for name, development_password in development_passwords.items()
        if os.getenv(name) == development_password
    ]
    short_passwords = [
        name
        for name in development_passwords
        if len(os.getenv(name, "")) < 12
    ]
    if unchanged_passwords or short_passwords:
        invalid_settings = sorted(set(unchanged_passwords + short_passwords))
        raise RuntimeError(
            "Replace demo passwords and use at least 12 characters for: "
            + ", ".join(invalid_settings)
        )

SESSION_SECRET_KEY = os.getenv(
    "SESSION_SECRET_KEY", "development-only-change-this-session-secret-key"
)

DEMO_PASSWORDS = {
    "admin": os.getenv("ADMIN_PASSWORD", "AdminDemo123!"),
    "analyst": os.getenv("ANALYST_PASSWORD", "AnalystDemo123!"),
    "viewer": os.getenv("VIEWER_PASSWORD", "ViewerDemo123!"),
}

ACCOUNTS = {
    "admin": {
        "username": os.getenv("ADMIN_USERNAME", "admin"),
        "role": "admin",
        "password": DEMO_PASSWORDS["admin"],
    },
    "analyst": {
        "username": os.getenv("ANALYST_USERNAME", "analyst"),
        "role": "analyst",
        "password": DEMO_PASSWORDS["analyst"],
    },
    "viewer": {
        "username": os.getenv("VIEWER_USERNAME", "viewer"),
        "role": "viewer",
        "password": DEMO_PASSWORDS["viewer"],
    },
}