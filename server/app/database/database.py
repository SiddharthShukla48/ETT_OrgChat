from app.core.database import get_db


def get_db_connection():
    """Backward-compatible alias for the shared SQLAlchemy session dependency."""
    yield from get_db()
