"""FastAPI router exposing the models. Mount it in the backend::

    from fastapi import FastAPI
    from alexandria_models.api import lifespan_for, router

    app = FastAPI(lifespan=lifespan_for("artifacts"))
    app.include_router(router, prefix="/ml")

or run it standalone for testing::

    ALEXANDRIA_ARTIFACTS=artifacts uvicorn alexandria_models.api:app --port 8001

Endpoints are plain ``def`` (not ``async def``) on purpose: FastAPI then runs them in its
thread pool, so slow model calls don't block the event loop.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import APIRouter, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .inference import AlexandriaModels

router = APIRouter(tags=["models"])


class SentencesRequest(BaseModel):
    level: int = Field(ge=1, description="difficulty level, 1 = easiest")
    n: int = Field(default=5, ge=1, le=50)
    word: str | None = Field(default=None, description="Hindi word the sentences must contain")
    grammar: str | None = Field(default=None, description="construction name, see GET /levels")
    mode: Literal["auto", "bank", "generate"] = "auto"
    exclude: list[str] = Field(default_factory=list, description="Hindi sentences already shown")


class DifficultyRequest(BaseModel):
    sentence: str = Field(min_length=1)


class EvaluateRequest(BaseModel):
    source: str = Field(min_length=1, description="sentence shown to the learner")
    answer: str = Field(description="learner's translation (Hindi may be romanized)")
    source_lang: Literal["hi", "en"] | None = None


class TransliterateRequest(BaseModel):
    text: str = Field(min_length=1)
    k: int = Field(default=3, ge=1, le=10)


def get_models(request: Request) -> AlexandriaModels:
    models = getattr(request.app.state, "models", None)
    if models is None:
        raise HTTPException(503, "models are not loaded")
    return models


def _call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    except RuntimeError as e:  # a model that is not loaded
        raise HTTPException(503, str(e)) from e


@router.get("/health")
def health(request: Request) -> dict:
    return get_models(request).status()


@router.get("/levels")
def levels(request: Request) -> dict:
    m = get_models(request)
    from .difficulty.features import CONSTRUCTIONS

    return {
        "num_levels": m.num_levels,
        "levels": m.levels(),
        "constructions": [{"name": c.name, "step": c.step, "description": c.description} for c in CONSTRUCTIONS],
    }


@router.post("/sentences")
def sentences(req: SentencesRequest, request: Request) -> dict:
    m = get_models(request)
    items = _call(m.sentences, req.level, req.n, req.word, req.grammar, req.mode, req.exclude)
    return {"level": req.level, "sentences": items}


@router.post("/difficulty")
def difficulty(req: DifficultyRequest, request: Request) -> dict:
    return _call(get_models(request).difficulty, req.sentence)


@router.post("/evaluate")
def evaluate(req: EvaluateRequest, request: Request) -> dict:
    return _call(get_models(request).evaluate, req.source, req.answer, req.source_lang)


@router.post("/transliterate")
def transliterate(req: TransliterateRequest, request: Request) -> dict:
    return {"readings": _call(get_models(request).transliterate, req.text, req.k)}


def lifespan_for(artifacts_dir: str, **load_kwargs):
    """Lifespan that loads the models once at startup into ``app.state.models``."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.models = AlexandriaModels.load(artifacts_dir, **load_kwargs)
        yield
        app.state.models = None

    return lifespan


def create_app(artifacts_dir: str | None = None, **load_kwargs) -> FastAPI:
    app = FastAPI(
        title="Alexandria models",
        lifespan=lifespan_for(artifacts_dir or os.environ.get("ALEXANDRIA_ARTIFACTS", "artifacts"), **load_kwargs),
    )
    app.include_router(router)
    return app


app = create_app()
