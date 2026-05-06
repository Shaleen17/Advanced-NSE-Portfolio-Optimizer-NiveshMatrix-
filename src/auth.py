"""MongoDB-backed authentication helpers for NiveshMatrix."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from pymongo import ASCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError, PyMongoError, ServerSelectionTimeoutError


PBKDF2_ITERATIONS = 210_000


@dataclass
class AuthResult:
    """Authentication operation result."""

    success: bool
    message: str
    user: dict | None = None


class AuthConfigError(RuntimeError):
    """Raised when MongoDB auth configuration is missing."""


def _secret_value(*keys: str, default: str | None = None) -> str | None:
    """Read a nested Streamlit secret value safely."""
    current = st.secrets
    try:
        for key in keys:
            current = current[key]
        return str(current)
    except Exception:
        return default


def get_mongo_settings() -> tuple[str, str, str]:
    """Return MongoDB URI, database name, and collection name."""
    uri = (
        _secret_value("mongo", "uri")
        or _secret_value("MONGODB_URI")
        or os.getenv("MONGODB_URI")
    )
    database = (
        _secret_value("mongo", "database")
        or os.getenv("MONGODB_DATABASE")
        or "niveshmatrix"
    )
    users_collection = (
        _secret_value("mongo", "users_collection")
        or os.getenv("MONGODB_USERS_COLLECTION")
        or "users"
    )
    if not uri:
        raise AuthConfigError("MongoDB URI is not configured.")
    return uri, database, users_collection


@st.cache_resource(show_spinner=False)
def get_users_collection() -> Collection:
    """Connect to MongoDB and return the users collection."""
    uri, database_name, collection_name = get_mongo_settings()
    client = MongoClient(uri, serverSelectionTimeoutMS=7000)
    client.admin.command("ping")
    collection = client[database_name][collection_name]
    collection.create_index([("email", ASCENDING)], unique=True)
    return collection


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    """Hash a password with PBKDF2-HMAC-SHA256."""
    if salt is None:
        salt = secrets.token_hex(32)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        PBKDF2_ITERATIONS,
    )
    return base64.b64encode(digest).decode("utf-8"), salt


def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    """Verify a password against its stored hash."""
    candidate_hash, _ = hash_password(password, salt)
    return hmac.compare_digest(candidate_hash, stored_hash)


def sanitize_user(user: dict) -> dict:
    """Return a user object without password fields."""
    return {
        "id": str(user.get("_id", "")),
        "email": user.get("email", ""),
        "name": user.get("name", ""),
        "created_at": user.get("created_at"),
        "last_login_at": user.get("last_login_at"),
    }


def create_user(name: str, email: str, password: str) -> AuthResult:
    """Create a new user in MongoDB."""
    clean_name = name.strip()
    clean_email = email.strip().lower()
    if len(clean_name) < 2:
        return AuthResult(False, "Please enter your full name.")
    if "@" not in clean_email or "." not in clean_email:
        return AuthResult(False, "Please enter a valid email address.")
    if len(password) < 8:
        return AuthResult(False, "Password must be at least 8 characters long.")

    password_hash, salt = hash_password(password)
    now = datetime.now(timezone.utc)
    document = {
        "name": clean_name,
        "email": clean_email,
        "password_hash": password_hash,
        "password_salt": salt,
        "created_at": now,
        "last_login_at": now,
    }

    try:
        collection = get_users_collection()
        result = collection.insert_one(document)
        document["_id"] = result.inserted_id
        return AuthResult(True, "Account created successfully.", sanitize_user(document))
    except DuplicateKeyError:
        return AuthResult(False, "An account with this email already exists.")
    except (ServerSelectionTimeoutError, PyMongoError):
        return AuthResult(
            False,
            "Unable to connect to the authentication database. Please check MongoDB settings.",
        )


def authenticate_user(email: str, password: str) -> AuthResult:
    """Authenticate a user with email and password."""
    clean_email = email.strip().lower()
    if not clean_email or not password:
        return AuthResult(False, "Please enter email and password.")

    try:
        collection = get_users_collection()
        user = collection.find_one({"email": clean_email})
        if not user:
            return AuthResult(False, "Invalid email or password.")
        stored_hash = user.get("password_hash")
        stored_salt = user.get("password_salt")
        if not stored_hash or not stored_salt:
            return AuthResult(False, "Invalid email or password.")
        if not verify_password(password, stored_hash, stored_salt):
            return AuthResult(False, "Invalid email or password.")
        collection.update_one(
            {"_id": user["_id"]},
            {"$set": {"last_login_at": datetime.now(timezone.utc)}},
        )
        user["last_login_at"] = datetime.now(timezone.utc)
        return AuthResult(True, "Login successful.", sanitize_user(user))
    except (ServerSelectionTimeoutError, PyMongoError):
        return AuthResult(
            False,
            "Unable to connect to the authentication database. Please check MongoDB settings.",
        )


def image_to_data_uri(path: Path) -> str:
    """Convert a local image to a browser data URI."""
    suffix = path.suffix.lower().replace(".", "")
    mime = "image/webp" if suffix == "webp" else f"image/{suffix or 'png'}"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"
