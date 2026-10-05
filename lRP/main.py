import logging
import secrets
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from http.cookies import SimpleCookie

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.sessions import SessionMiddleware

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

from app.auth import authenticate
from app.config import APP_ENV, SESSION_SECRET_KEY
from app.dependencies import require_roles, require_user
from app.model_service import ModelService
from app.schemas import PredictionInput

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.model_service = ModelService.load()
    if application.state.model_service is None:
        logger.warning("Trained model is missing; run python train_model.py")
    yield
    application.state.model_service = None


app = FastAPI(
    title="Logistic Regression Intelligence Platform",
    description="A secure educational dashboard for an advertising response model.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET_KEY,
    session_cookie="lr_platform_session",
    max_age=None,
    same_site="lax",
    https_only=APP_ENV == "production",
)
app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static",
)
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.middleware("http")
async def remember_selected_session(request: Request, call_next):
    response = await call_next(request)
    if request.scope.get("session", {}).get("remember_me"):
        cookies = SimpleCookie()
        cookies.load(response.headers.get("set-cookie", ""))
        session_cookie = cookies.get("lr_platform_session")
        if session_cookie is not None:
            session_cookie["max-age"] = str(30 * 24 * 60 * 60)
            response.headers["set-cookie"] = session_cookie.OutputString()
    return response

NAVIGATION = [
    ("Dashboard", "/dashboard", {"admin", "analyst", "viewer"}),
    ("Prediction", "/prediction", {"admin", "analyst", "viewer"}),
    ("Evaluation", "/evaluation", {"admin", "analyst"}),
    ("About", "/about", {"admin", "analyst", "viewer"}),
    ("Profile", "/profile", {"admin", "analyst", "viewer"}),
]


def csrf_token(request: Request) -> str:
    token = request.session.get("csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        request.session["csrf_token"] = token
    return token


def render_page(
    request: Request,
    template_name: str,
    context: dict | None = None,
    status_code: int = 200,
) -> Response:
    values = dict(context or {})
    user = getattr(request.state, "user", None)
    values.update(
        request=request,
        user=user,
        csrf_token=csrf_token(request),
        navigation=[item for item in NAVIGATION if user and user["role"] in item[2]],
    )
    return templates.TemplateResponse(
        request=request,
        name=template_name,
        context=values,
        status_code=status_code,
    )


def validate_csrf(request: Request, submitted: str | None) -> bool:
    expected = request.session.get("csrf_token", "")
    return bool(expected and submitted and secrets.compare_digest(expected, submitted))


def model_service(request: Request) -> ModelService | None:
    return request.app.state.model_service


def role_for_target(metadata: dict, class_value: int) -> str:
    if metadata.get("target_name") == "Clicked on Ad":
        return "Likely to Click" if class_value == 1 else "Unlikely to Click"
    labels = metadata.get("class_labels", {})
    return str(labels.get(str(class_value), class_value))


@app.get("/", include_in_schema=False)
async def home(request: Request):
    if request.session.get("username"):
        return RedirectResponse("/dashboard", status_code=303)
    return RedirectResponse("/login", status_code=303)


@app.get("/login", include_in_schema=False)
async def login_page(request: Request):
    if request.session.get("username"):
        return RedirectResponse("/dashboard", status_code=303)
    service = model_service(request)
    return render_page(
        request,
        "login.html",
        {
            "error": None,
            "username": "",
            "metadata": service.metadata if service else {},
        },
    )


@app.post("/login", include_in_schema=False)
async def login_submit(request: Request):
    form = await request.form()
    username = str(form.get("username", "")).strip()
    password = str(form.get("password", ""))
    service = model_service(request)
    page_context = {
        "username": username,
        "metadata": service.metadata if service else {},
    }
    if not validate_csrf(request, str(form.get("csrf_token", ""))):
        page_context["error"] = "Your form expired. Please try again."
        return render_page(
            request,
            "login.html",
            page_context,
            status_code=400,
        )
    account = authenticate(username, password)
    if account is None:
        page_context["error"] = "Username or password was not recognized."
        return render_page(
            request,
            "login.html",
            page_context,
            status_code=401,
        )
    request.session.clear()
    request.session["username"] = account["username"]
    request.session["remember_me"] = form.get("remember_me") == "yes"
    request.session["csrf_token"] = secrets.token_urlsafe(32)
    return RedirectResponse("/dashboard", status_code=303)


@app.post("/logout", include_in_schema=False)
async def logout(request: Request):
    form = await request.form()
    if not validate_csrf(request, str(form.get("csrf_token", ""))):
        raise HTTPException(status_code=400, detail="Invalid security token")
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


@app.get("/dashboard", include_in_schema=False)
async def dashboard(request: Request, user: dict = Depends(require_user)):
    service = model_service(request)
    metadata = service.metadata if service else {}
    return render_page(
        request,
        "dashboard.html",
        {
            "metadata": metadata,
            "recent_predictions": request.session.get("recent_predictions", []),
            "model_loaded": service is not None,
            "active_page": "Dashboard",
        },
    )


@app.get("/prediction", include_in_schema=False)
async def prediction_page(request: Request, user: dict = Depends(require_user)):
    service = model_service(request)
    if service is None:
        return render_page(request, "missing_model.html", status_code=503)
    fields = service.metadata["features"]
    return render_page(
        request,
        "prediction.html",
        {"features": fields, "values": [""] * len(fields), "errors": {}, "result": None},
    )


@app.post("/prediction", include_in_schema=False)
async def predict(request: Request, user: dict = Depends(require_user)):
    form = await request.form()
    service = model_service(request)
    if service is None:
        return render_page(request, "missing_model.html", status_code=503)
    fields = service.metadata["features"]
    raw_values = [str(value).strip() for value in form.getlist("features")]
    submitted_token = str(form.get("csrf_token", ""))
    if not validate_csrf(request, submitted_token):
        errors = {"form": "Your form expired. Reload the page and try again."}
        return render_page(
            request,
            "prediction.html",
            {"features": fields, "values": raw_values, "errors": errors, "result": None},
            status_code=400,
        )

    errors: dict[str, str] = {}
    try:
        payload = PredictionInput(features=[float(value) for value in raw_values])
    except (ValueError, TypeError):
        payload = None
        errors["form"] = "Enter a valid number for every model feature."

    if payload is not None and len(payload.features) != len(fields):
        errors["form"] = f"Enter all {len(fields)} values in the order shown."
    if payload is not None and not errors:
        for field, value in zip(fields, payload.features):
            if value < field["min"] or value > field["max"]:
                errors[field["name"]] = (
                    f"Use a value from {field['min']:g} to {field['max']:g}, "
                    "based on the training dataset."
                )
            if field["name"] == "Male" and value not in (0, 1):
                errors[field["name"]] = "Choose 0 or 1."

    if errors:
        padded_values = (raw_values + [""] * len(fields))[: len(fields)]
        return render_page(
            request,
            "prediction.html",
            {"features": fields, "values": padded_values, "errors": errors, "result": None},
            status_code=422,
        )

    probabilities = service.predict_probabilities(payload.features)
    predicted_class = service.predict(payload.features)
    confidence = max(probabilities.values())
    result = {
        "class": predicted_class,
        "label": role_for_target(service.metadata, predicted_class),
        "probability_0": probabilities.get(0, 0.0),
        "probability_1": probabilities.get(1, 0.0),
        "confidence": confidence,
    }
    history = request.session.get("recent_predictions", [])
    history.insert(
        0,
        {
            "label": result["label"],
            "confidence": round(confidence * 100, 1),
            "created_at": datetime.now().astimezone().strftime("%b %d, %H:%M"),
        },
    )
    request.session["recent_predictions"] = history[:5]
    return render_page(
        request,
        "prediction.html",
        {"features": fields, "values": [str(value) for value in payload.features], "errors": {}, "result": result},
    )


@app.post("/api/predict")
async def api_predict(
    payload: PredictionInput,
    request: Request,
    csrf: str | None = Header(default=None, alias="X-CSRF-Token"),
    user: dict = Depends(require_user),
):
    if not validate_csrf(request, csrf):
        return JSONResponse({"detail": "Invalid security token"}, status_code=403)
    service = model_service(request)
    if service is None:
        return JSONResponse({"detail": "Trained model is unavailable"}, status_code=503)
    fields = service.metadata["features"]
    if len(payload.features) != len(fields):
        return JSONResponse(
            {"detail": f"Expected {len(fields)} feature values in the displayed order"},
            status_code=422,
        )
    for field, value in zip(fields, payload.features):
        if value < field["min"] or value > field["max"]:
            return JSONResponse(
                {"detail": f"{field['name']} must be between {field['min']:g} and {field['max']:g}"},
                status_code=422,
            )
        if field["name"] == "Male" and value not in (0, 1):
            return JSONResponse({"detail": "Male must be 0 or 1"}, status_code=422)

    probabilities = service.predict_probabilities(payload.features)
    predicted_class = service.predict(payload.features)
    confidence = max(probabilities.values())
    return {
        "predicted_class": predicted_class,
        "label": role_for_target(service.metadata, predicted_class),
        "probability_class_0": probabilities.get(0, 0.0),
        "probability_class_1": probabilities.get(1, 0.0),
        "confidence": confidence,
    }


@app.get("/performance", include_in_schema=False)
async def performance(
    request: Request,
    user: dict = Depends(require_roles("admin", "analyst")),
):
    service = model_service(request)
    return render_page(
        request,
        "performance.html",
        {"metadata": service.metadata if service else {}, "model_loaded": service is not None},
    )


@app.get("/evaluation", include_in_schema=False)
async def evaluation(
    request: Request,
    user: dict = Depends(require_roles("admin", "analyst")),
):
    service = model_service(request)
    return render_page(
        request,
        "evaluation.html",
        {"metadata": service.metadata if service else {}, "model_loaded": service is not None},
    )


@app.get("/analytics", include_in_schema=False)
async def analytics(
    request: Request,
    user: dict = Depends(require_roles("admin", "analyst")),
):
    service = model_service(request)
    return render_page(
        request,
        "analytics.html",
        {"metadata": service.metadata if service else {}, "model_loaded": service is not None},
    )


@app.get("/insights", include_in_schema=False)
async def insights(
    request: Request,
    user: dict = Depends(require_roles("admin", "analyst")),
):
    service = model_service(request)
    return render_page(
        request,
        "insights.html",
        {"metadata": service.metadata if service else {}, "model_loaded": service is not None},
    )


@app.get("/about", include_in_schema=False)
async def about(request: Request, user: dict = Depends(require_user)):
    service = model_service(request)
    return render_page(
        request,
        "about.html",
        {"metadata": service.metadata if service else {}},
    )


@app.get("/profile", include_in_schema=False)
async def profile(request: Request, user: dict = Depends(require_user)):
    permissions = [
        label for label, _, roles in NAVIGATION if user["role"] in roles
    ]
    return render_page(
        request,
        "profile.html",
        {"account": user, "permissions": permissions},
    )


@app.get("/access-denied", include_in_schema=False)
async def access_denied(request: Request):
    return render_page(request, "access_denied.html", status_code=403)


@app.get("/model-unavailable", include_in_schema=False)
async def model_unavailable(request: Request, user: dict = Depends(require_user)):
    return render_page(request, "missing_model.html", status_code=503)


@app.get("/health")
async def health(request: Request):
    loaded = model_service(request) is not None
    return {
        "status": "healthy" if loaded else "degraded",
        "application": app.title,
        "model_loaded": loaded,
    }


@app.get("/api/health")
async def api_health(request: Request):
    return await health(request)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    if exc.status_code in (302, 303, 307, 308) and exc.headers and "location" in exc.headers:
        return RedirectResponse(exc.headers["location"], status_code=exc.status_code)
    if exc.status_code == 403:
        return render_page(request, "access_denied.html", status_code=403)
    if exc.status_code == 404:
        return render_page(request, "404.html", status_code=404)
    if exc.status_code == 503:
        return render_page(request, "missing_model.html", status_code=503)
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers)


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    logger.exception("Unhandled application error")
    return render_page(request, "500.html", status_code=500)