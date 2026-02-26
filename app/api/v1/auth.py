"""
Authentication and authorization endpoints for the API.
"""

import uuid
from typing import List

from fastapi import (
    APIRouter,
    Depends,
    Form,
    HTTPException,
    Request,
)
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import jwt, JWTError

from app.core.config import settings
from app.core.limiter import limiter
from app.core.logging import bind_context, logger
from app.models.session import Session
from app.models.user import User
from app.schemas.auth import (
    SessionResponse,
    TokenResponse,
    UserCreate,
    UserResponse,
)
from app.services.database import DatabaseService
from app.utils.auth import create_access_token
from app.utils.sanitization import (
    sanitize_email,
    sanitize_string,
    validate_password_strength,
)

router = APIRouter()
security = HTTPBearer()
db_service = DatabaseService()


# USER AUTH

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> User:
    try:
        token = sanitize_string(credentials.credentials)

        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )

        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")

        user = await db_service.get_user(int(user_id))
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        bind_context(user_id=user.id)
        return user

    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid token format")



# SESSION AUTH

async def get_current_session(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Session:
    try:
        token = sanitize_string(credentials.credentials)

        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )

        session_id = payload.get("sub")
        if not session_id:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")

        session_id = sanitize_string(session_id)

        session = await db_service.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")

        bind_context(user_id=session.user_id)
        return session

    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid token format")


# REGISTER

@router.post("/register", response_model=UserResponse)
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["register"][0])
async def register_user(request: Request, user_data: UserCreate):
    sanitized_email = sanitize_email(user_data.email)
    password = user_data.password.get_secret_value()
    validate_password_strength(password)

    if await db_service.get_user_by_email(sanitized_email):
        raise HTTPException(status_code=400, detail="Email already registered")

    user = await db_service.create_user(
        email=sanitized_email,
        password=User.hash_password(password),
    )

    token = create_access_token(str(user.id))

    return UserResponse(
        id=user.id,
        email=user.email,
        token=token,
    )



# LOGIN

@router.post("/login", response_model=TokenResponse)
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["login"][0])
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    grant_type: str = Form(default="password"),
):
    username = sanitize_string(username)
    password = sanitize_string(password)

    if grant_type != "password":
        raise HTTPException(status_code=400, detail="Unsupported grant type")

    user = await db_service.get_user_by_email(username)
    if not user or not user.verify_password(password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    token = create_access_token(str(user.id))

    return TokenResponse(
        access_token=token.access_token,
        token_type="bearer",
        expires_at=token.expires_at,
    )



# CREATE SESSION

@router.post("/session", response_model=SessionResponse)
async def create_session(user: User = Depends(get_current_user)):
    session_id = str(uuid.uuid4())

    session = await db_service.create_session(session_id, user.id)

    token = create_access_token(session_id)

    logger.info(
        "session_created",
        session_id=session_id,
        user_id=user.id,
    )

    return SessionResponse(
        session_id=session_id,
        name=session.name,
        token=token,
    )



# UPDATE SESSION NAME

@router.patch("/session/{session_id}/name", response_model=SessionResponse)
async def update_session_name(
    session_id: str,
    name: str = Form(...),
    current_session: Session = Depends(get_current_session),
):
    session_id = sanitize_string(session_id)
    name = sanitize_string(name)

    if session_id != current_session.id:
        raise HTTPException(status_code=403, detail="Cannot modify other sessions")

    session = await db_service.update_session_name(session_id, name)
    token = create_access_token(session_id)

    return SessionResponse(
        session_id=session_id,
        name=session.name,
        token=token,
    )



# DELETE SESSION

@router.delete("/session/{session_id}")
async def delete_session(
    session_id: str,
    current_session: Session = Depends(get_current_session),
):
    session_id = sanitize_string(session_id)

    if session_id != current_session.id:
        raise HTTPException(status_code=403, detail="Cannot delete other sessions")

    await db_service.delete_session(session_id)

    logger.info(
        "session_deleted",
        session_id=session_id,
        user_id=current_session.user_id,
    )



# GET USER SESSIONS

@router.get("/sessions", response_model=List[SessionResponse])
async def get_user_sessions(user: User = Depends(get_current_user)):
    sessions = await db_service.get_user_sessions(user.id)

    return [
        SessionResponse(
            session_id=session.id,
            name=session.name,
            token=create_access_token(session.id),
        )
        for session in sessions
    ]