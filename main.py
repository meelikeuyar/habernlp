"""HaberNLP — Entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from loguru import logger

from config.settings import API_HOST, API_PORT, LOG_LEVEL
from src.database import init_db
from src.api.routes import router
from src.scheduler.jobs import start_scheduler, stop_scheduler


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("🚀 HaberNLP starting up...")
    init_db()
    start_scheduler()
    yield
    stop_scheduler()
    logger.info("HaberNLP shut down.")


app = FastAPI(
    title="HaberNLP",
    description="Turkish News Intelligence Platform",
    version="2.0.0",
    lifespan=lifespan,
)

app.include_router(router, prefix="/api")


@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Serve the dashboard."""
    from pathlib import Path
    html_path = Path(__file__).parent / "frontend" / "templates" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import uvicorn

    logger.add("data/habernlp.log", rotation="10 MB", level=LOG_LEVEL)
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=True)
