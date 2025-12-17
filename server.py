"""
Onyx Inference API Server

FastAPI server with streaming SSE support for Onyx model inference.

Usage:
    uvicorn server:app --host 0.0.0.0 --port 8000

    # Or with auto-reload for development:
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import sys
import time
import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any, AsyncGenerator, Tuple
from contextlib import asynccontextmanager

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

# Add model path to sys.path
MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "onyx"
sys.path.insert(0, str(MODEL_DIR))

from onyx_model import Onyx, OnyxConfig
from onyx_inference import load_model as load_onyx_model

try:
    from transformers import AutoTokenizer
except ImportError:
    raise RuntimeError("transformers not installed. Run: pip install transformers")


# =============================================================================
# Configuration
# =============================================================================

class ServerConfig:
    checkpoint_path: str = os.environ.get(
        "ONYX_CHECKPOINT_PATH",
        str(MODEL_DIR / "checkpoints" / "checkpoint_step_1000.pt"),
    )
    model_config_path: str = os.environ.get(
        "ONYX_MODEL_CONFIG_PATH",
        str(MODEL_DIR / "models" / "150m" / "config.json"),
    )
    tokenizer_name: str = os.environ.get("ONYX_TOKENIZER_NAME", "NousResearch/Hermes-2-Pro-Llama-3-8B")
    device: str = os.environ.get("ONYX_DEVICE", "auto")  # auto, cuda, mps, cpu
    dtype: str = os.environ.get("ONYX_DTYPE", "float32")  # float32, float16, bfloat16
    max_seq_len: int = 4096
    default_max_tokens: int = 512
    default_temperature: float = 0.8
    default_top_p: float = 0.9
    default_top_k: int = 50
    allow_local_paths: bool = os.environ.get("ONYX_ALLOW_LOCAL_PATHS", "0") == "1"
    allowed_path_prefixes: str = os.environ.get("ONYX_ALLOWED_PATH_PREFIXES", str(MODEL_DIR))
    model_cache_size: int = int(os.environ.get("ONYX_MODEL_CACHE_SIZE", "2"))
    skip_startup_load: bool = os.environ.get("ONYX_SKIP_LOAD", "0") == "1"
    registry_url: Optional[str] = os.environ.get("ONYX_REGISTRY_URL") or None
    registry_api_key: Optional[str] = os.environ.get("ONYX_REGISTRY_API_KEY") or None
    artifact_cache_url: Optional[str] = os.environ.get("ONYX_ARTIFACT_CACHE_URL") or None
    artifact_cache_api_key: Optional[str] = os.environ.get("ONYX_ARTIFACT_CACHE_API_KEY") or None
    min_registry_status: str = os.environ.get("ONYX_MIN_REGISTRY_STATUS", "experimental")


config = ServerConfig()


# =============================================================================
# Global State
# =============================================================================

class ModelState:
    # Default (startup) bundle
    model: Optional[Onyx] = None
    tokenizer: Any = None
    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None
    config: Optional[OnyxConfig] = None

    manager: Optional["ModelManager"] = None

    # Session memory states keyed by model_key -> session_id
    sessions: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}


state = ModelState()


# =============================================================================
# Model Loading
# =============================================================================

def _select_device() -> torch.device:
    if config.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(config.device)


def _select_dtype() -> torch.dtype:
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if config.dtype not in dtype_map:
        raise RuntimeError(f"Unsupported dtype: {config.dtype}")
    return dtype_map[config.dtype]


def _allowed_prefixes() -> List[Path]:
    prefixes: List[Path] = []
    for raw in (config.allowed_path_prefixes or "").split(","):
        raw = raw.strip()
        if raw:
            prefixes.append(Path(raw).expanduser().resolve())
    if not prefixes:
        prefixes = [MODEL_DIR.resolve()]
    return prefixes


def _is_allowed_local_path(p: Path) -> bool:
    rp = p.expanduser().resolve()
    for base in _allowed_prefixes():
        try:
            rp.relative_to(base)
            return True
        except Exception:
            continue
    return False


def _normalize_local_path(raw: str) -> Path:
    s = (raw or "").strip()
    if "://" in s and not s.startswith("file://"):
        raise HTTPException(status_code=400, detail=f"Remote URI not supported here: {raw}")
    if s.startswith("file://"):
        s = s[len("file://") :]
    return Path(s).expanduser().resolve()


@dataclass(frozen=True)
class ModelSelector:
    model_id: Optional[int] = None
    artifact_uri: Optional[str] = None
    checkpoint_path: Optional[str] = None
    model_config_path: Optional[str] = None
    checkpoint_sha256: Optional[str] = None
    checkpoint_size_bytes: Optional[int] = None
    config_sha256: Optional[str] = None
    config_size_bytes: Optional[int] = None
    registry_status: Optional[str] = None


@dataclass(frozen=True)
class ModelKey:
    checkpoint_path: str
    model_config_path: Optional[str]
    device: str
    dtype: str


@dataclass
class ModelBundle:
    key: ModelKey
    model: Onyx
    tokenizer: Any
    device: torch.device
    dtype: torch.dtype
    config: OnyxConfig


class ModelManager:
    def __init__(self, *, tokenizer: Any, device: torch.device, dtype: torch.dtype):
        self._tokenizer = tokenizer
        self._device = device
        self._dtype = dtype
        self._lock = asyncio.Lock()
        self._inflight: Dict[ModelKey, asyncio.Future[ModelBundle]] = {}
        self._cache: "OrderedDict[ModelKey, ModelBundle]" = OrderedDict()

    def _key_id(self, key: ModelKey) -> str:
        return f"{key.device}:{key.dtype}:{key.checkpoint_path}|{key.model_config_path or ''}"

    async def _fetch_registry_model(self, model_id: int) -> Optional[dict]:
        if not config.registry_url:
            return None
        import httpx

        headers: Dict[str, str] = {}
        if config.registry_api_key:
            headers["X-API-Key"] = config.registry_api_key
        url = config.registry_url.rstrip("/") + f"/models/{model_id}"
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(url, headers=headers)
            r.raise_for_status()
            data = r.json()
        return data if isinstance(data, dict) else None

    def _registry_status_ok(self, status: Optional[str]) -> bool:
        if status is None:
            return True
        order = {"experimental": 0, "staging": 1, "production": 2, "archived": -1}
        min_status = (config.min_registry_status or "experimental").strip().lower()
        if min_status not in order:
            min_status = "experimental"
        s = status.strip().lower()
        if s not in order:
            return True
        if s == "archived":
            return False
        return order[s] >= order[min_status]

    async def _resolve_via_artifact_cache(self, *, artifact_uri: str, sha256: str, size_bytes: int) -> str:
        if not config.artifact_cache_url:
            raise HTTPException(
                status_code=400,
                detail="Remote artifacts require ONYX_ARTIFACT_CACHE_URL (artifact cache service)",
            )
        import httpx

        base = config.artifact_cache_url.rstrip("/")
        url = base if base.endswith("/resolve") else base + "/resolve"
        headers: Dict[str, str] = {}
        if config.artifact_cache_api_key:
            headers["X-API-Key"] = config.artifact_cache_api_key
        payload = {"artifact_uri": artifact_uri, "sha256": sha256, "size_bytes": int(size_bytes)}
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(url, json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
        if not isinstance(data, dict) or not isinstance(data.get("local_path"), str):
            raise HTTPException(status_code=502, detail="Artifact cache returned invalid response")
        return data["local_path"]

    async def _resolve_selector(self, selector: ModelSelector) -> ModelSelector:
        # If model_id is provided but paths are missing, optionally resolve via registry.
        if selector.model_id is None:
            return selector
        if selector.checkpoint_path or selector.artifact_uri or selector.model_config_path:
            return selector
        data = await self._fetch_registry_model(selector.model_id)
        if not data:
            return selector
        if not self._registry_status_ok(data.get("status") if isinstance(data.get("status"), str) else None):
            raise HTTPException(status_code=403, detail="Model not allowed by ONYX_MIN_REGISTRY_STATUS")
        # model-registry may expose local paths only in dev.
        ckpt = data.get("local_checkpoint_path") or data.get("checkpoint_path") or data.get("artifact_uri")
        cfgp = data.get("local_config_path") or data.get("config_path")
        return ModelSelector(
            model_id=selector.model_id,
            artifact_uri=data.get("artifact_uri") if isinstance(data.get("artifact_uri"), str) else None,
            checkpoint_path=ckpt if isinstance(ckpt, str) else None,
            model_config_path=cfgp if isinstance(cfgp, str) else None,
            checkpoint_sha256=data.get("checkpoint_sha256") if isinstance(data.get("checkpoint_sha256"), str) else None,
            checkpoint_size_bytes=int(data["checkpoint_size_bytes"]) if isinstance(data.get("checkpoint_size_bytes"), int) else None,
            config_sha256=data.get("config_sha256") if isinstance(data.get("config_sha256"), str) else None,
            config_size_bytes=int(data["config_size_bytes"]) if isinstance(data.get("config_size_bytes"), int) else None,
            registry_status=data.get("status") if isinstance(data.get("status"), str) else None,
        )

    async def _resolve_key(self, selector: ModelSelector) -> ModelKey:
        selector = await self._resolve_selector(selector)

        ckpt_path = None
        cfg_path = None

        if selector.model_config_path:
            cfg_path = selector.model_config_path

        # Explicit checkpoint path takes precedence (dev-only).
        if selector.checkpoint_path:
            ckpt_path = selector.checkpoint_path
        elif selector.artifact_uri:
            ckpt_path = selector.artifact_uri

        if ckpt_path:
            # Remote artifacts: resolve via artifact cache using registry-provided sha/size (or request-provided).
            if ckpt_path.startswith("s3://"):
                if selector.checkpoint_sha256 is None or selector.checkpoint_size_bytes is None:
                    raise HTTPException(status_code=400, detail="Remote checkpoint requires checkpoint_sha256 and checkpoint_size_bytes")
                ckpt_path = await self._resolve_via_artifact_cache(
                    artifact_uri=ckpt_path,
                    sha256=selector.checkpoint_sha256,
                    size_bytes=selector.checkpoint_size_bytes,
                )
            else:
                # Local path override (dev-only).
                if not config.allow_local_paths:
                    raise HTTPException(
                        status_code=400,
                        detail="Checkpoint path overrides are disabled (set ONYX_ALLOW_LOCAL_PATHS=1) or use model_id with artifact cache",
                    )
                p = _normalize_local_path(ckpt_path)
                if not p.exists():
                    raise HTTPException(status_code=400, detail=f"Checkpoint not found: {p}")
                if not _is_allowed_local_path(p):
                    raise HTTPException(status_code=400, detail=f"Checkpoint path not allowed: {p}")
                ckpt_path = str(p)
        else:
            ckpt_path = config.checkpoint_path

        if cfg_path:
            if cfg_path.startswith("s3://"):
                if selector.config_sha256 is None or selector.config_size_bytes is None:
                    raise HTTPException(status_code=400, detail="Remote config requires config_sha256 and config_size_bytes")
                cfg_path = await self._resolve_via_artifact_cache(
                    artifact_uri=cfg_path,
                    sha256=selector.config_sha256,
                    size_bytes=selector.config_size_bytes,
                )
            else:
                if not config.allow_local_paths:
                    raise HTTPException(
                        status_code=400,
                        detail="Config path overrides are disabled (set ONYX_ALLOW_LOCAL_PATHS=1) or use model_id with artifact cache",
                    )
                p = _normalize_local_path(cfg_path)
                if not p.exists():
                    raise HTTPException(status_code=400, detail=f"Config not found: {p}")
                if not _is_allowed_local_path(p):
                    raise HTTPException(status_code=400, detail=f"Config path not allowed: {p}")
                cfg_path = str(p)
        else:
            cfg_path = config.model_config_path

        return ModelKey(
            checkpoint_path=str(ckpt_path),
            model_config_path=str(cfg_path) if cfg_path else None,
            device=str(self._device),
            dtype=str(self._dtype).replace("torch.", ""),
        )

    def _load_bundle_sync(self, key: ModelKey) -> ModelBundle:
        model, model_cfg = load_onyx_model(
            checkpoint_path=key.checkpoint_path,
            tokenizer=self._tokenizer,
            device=self._device,
            dtype=self._dtype,
            model_config_path=key.model_config_path,
        )
        return ModelBundle(
            key=key,
            model=model,
            tokenizer=self._tokenizer,
            device=self._device,
            dtype=self._dtype,
            config=model_cfg,
        )

    async def get_bundle(self, selector: ModelSelector) -> Tuple[ModelBundle, str]:
        key = await self._resolve_key(selector)

        async with self._lock:
            if key in self._cache:
                bundle = self._cache.pop(key)
                self._cache[key] = bundle
                return bundle, self._key_id(key)

            fut = self._inflight.get(key)
            if fut is None:
                fut = asyncio.get_running_loop().create_future()
                self._inflight[key] = fut
                should_load = True
            else:
                should_load = False

        if should_load:
            try:
                bundle = await asyncio.to_thread(self._load_bundle_sync, key)
                async with self._lock:
                    self._cache[key] = bundle
                    while len(self._cache) > max(1, int(config.model_cache_size)):
                        self._cache.popitem(last=False)
                    fut = self._inflight.pop(key, None)
                    if fut is not None and not fut.done():
                        fut.set_result(bundle)
                return bundle, self._key_id(key)
            except Exception as e:
                async with self._lock:
                    fut = self._inflight.pop(key, None)
                    if fut is not None and not fut.done():
                        fut.set_exception(e)
                raise

        bundle = await fut
        return bundle, self._key_id(key)

    async def list_cached(self) -> List[str]:
        async with self._lock:
            return [self._key_id(k) for k in self._cache.keys()]

    async def clear_cache(self) -> int:
        async with self._lock:
            n = len(self._cache)
            self._cache.clear()
            self._inflight.clear()
            return n


def _extract_selector(req: Any) -> ModelSelector:
    return ModelSelector(
        model_id=getattr(req, "model_id", None),
        artifact_uri=getattr(req, "artifact_uri", None),
        checkpoint_path=getattr(req, "checkpoint_path", None),
        model_config_path=getattr(req, "model_config_path", None),
    )


# =============================================================================
# Request/Response Models
# =============================================================================

class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    prompt: str
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)
    stream: bool = Field(default=False)
    session_id: Optional[str] = Field(default=None, description="Session ID for memory persistence")
    system_prompt: Optional[str] = Field(default=None)
    stop_sequences: Optional[List[str]] = Field(default=None)
    # Model selection (eval-service compatibility: accepts model_id + artifact_uri)
    model_id: Optional[int] = Field(default=None)
    artifact_uri: Optional[str] = Field(default=None)
    checkpoint_path: Optional[str] = Field(default=None, description="Dev-only local checkpoint path override")
    model_config_path: Optional[str] = Field(default=None, description="Dev-only local model config path override")
    eval: Optional[Dict[str, Any]] = Field(default=None, description="Opaque eval metadata (ignored by server)")


class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int
    tokens_per_second: float
    session_id: Optional[str] = None


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    messages: List[ChatMessage]
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)
    stream: bool = Field(default=False)
    session_id: Optional[str] = Field(default=None)
    system_prompt: Optional[str] = Field(default=None)
    model_id: Optional[int] = Field(default=None)
    artifact_uri: Optional[str] = Field(default=None)
    checkpoint_path: Optional[str] = Field(default=None)
    model_config_path: Optional[str] = Field(default=None)
    eval: Optional[Dict[str, Any]] = Field(default=None)


class ModelInfo(BaseModel):
    name: str
    parameters: int
    d_model: int
    n_layers: int
    n_heads: int
    device: str
    dtype: str


# =============================================================================
# Sampling
# =============================================================================

def sample_token(
    logits: torch.Tensor,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
    generated_tokens: Optional[Any] = None,
) -> torch.Tensor:
    """Sample next token"""

    if repetition_penalty != 1.0 and generated_tokens is not None:
        if isinstance(generated_tokens, set):
            token_ids = generated_tokens
        elif torch.is_tensor(generated_tokens):
            token_ids = set(int(x) for x in generated_tokens.flatten().tolist())
        else:
            token_ids = set(int(x) for x in generated_tokens)
        for token_id in token_ids:
            if 0 <= token_id < logits.size(-1):
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty

    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = float('-inf')

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


# =============================================================================
# Generation
# =============================================================================

async def generate_tokens(
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    bundle: ModelBundle,
    model_key: str,
    session_id: Optional[str] = None,
    stop_sequences: Optional[List[str]] = None,
) -> AsyncGenerator[tuple[str, bool], None]:
    """Generate tokens, yielding each one"""

    model = bundle.model
    tokenizer = bundle.tokenizer
    device = bundle.device

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    B, S = input_ids.shape

    # Get or create memory states
    if session_id:
        per_model = state.sessions.setdefault(model_key, {})
        if session_id in per_model:
            memory_states = per_model[session_id]
        else:
            memory_states = model.init_memory_states(B, device, bundle.dtype)
    else:
        memory_states = model.init_memory_states(B, device, bundle.dtype)

    # Get stop token IDs
    stop_token_ids = [tokenizer.eos_token_id] if getattr(tokenizer, "eos_token_id", None) else []
    for token in ["<|eot_id|>", "<|end|>", "</s>", "<|im_end|>"]:
        if hasattr(tokenizer, "convert_tokens_to_ids") and hasattr(tokenizer, "unk_token_id"):
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id != tokenizer.unk_token_id:
                stop_token_ids.append(token_id)

    # Process prompt
    with torch.no_grad():
        outputs = model(
            input_ids,
            memory_states=memory_states,
            update_memories=True,
            inference_mode=True,
            position_offset=0,
        )
        memory_states = outputs["memory_states"]
        kv_cache = outputs.get("kv_cache")
        position_offset = S

    seen_token_ids: set[int] = set()
    generated_text = ""

    for _ in range(max_tokens):
        with torch.no_grad():
            logits = outputs["logits"][:, -1:, :]

            next_token = sample_token(
                logits.squeeze(1),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                generated_tokens=(seen_token_ids if repetition_penalty != 1.0 else None),
            )

            if repetition_penalty != 1.0:
                seen_token_ids.add(int(next_token.item()))
            token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
            generated_text += token_text

            # Check stop sequences
            should_stop = next_token.item() in stop_token_ids
            if stop_sequences:
                for seq in stop_sequences:
                    if generated_text.endswith(seq):
                        should_stop = True
                        break

            yield token_text, should_stop

            if should_stop:
                break

            outputs = model(
                next_token,
                memory_states=memory_states,
                update_memories=True,
                inference_mode=True,
                kv_cache=kv_cache,
                position_offset=position_offset,
            )
            memory_states = outputs["memory_states"]
            kv_cache = outputs.get("kv_cache")
            position_offset += 1

        # Yield control to event loop
        await asyncio.sleep(0)

    # Save session memory
    if session_id:
        state.sessions.setdefault(model_key, {})[session_id] = memory_states


# =============================================================================
# API Endpoints
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup"""
    device = _select_device()
    dtype = _select_dtype()
    print(f"Device: {device} | Dtype: {dtype}")

    print(f"Loading tokenizer: {config.tokenizer_name}")
    state.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, trust_remote_code=True)
    state.device = device
    state.dtype = dtype
    state.manager = ModelManager(tokenizer=state.tokenizer, device=device, dtype=dtype)

    if not config.skip_startup_load:
        print(f"Preloading default checkpoint: {config.checkpoint_path}")
        bundle, key_id = await state.manager.get_bundle(ModelSelector())
        state.model = bundle.model
        state.config = bundle.config
        state.sessions.setdefault(key_id, {})
        print(f"Model loaded: {bundle.model.get_num_params():,} parameters")
    yield
    # Cleanup on shutdown
    state.model = None
    state.tokenizer = None
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


app = FastAPI(
    title="Onyx Inference API",
    description="API for Onyx language model inference with streaming support",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"status": "ok", "model": "onyx"}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "manager_ready": state.manager is not None,
        "default_model_loaded": state.model is not None,
    }


@app.get("/model", response_model=ModelInfo)
async def model_info():
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfo(
        name=state.config.name if hasattr(state.config, 'name') else "onyx",
        parameters=state.model.get_num_params(),
        d_model=state.config.d_model,
        n_layers=state.config.n_layers,
        n_heads=state.config.n_heads,
        device=str(state.device),
        dtype=str(state.dtype),
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text from prompt"""
    if state.manager is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    bundle, model_key = await state.manager.get_bundle(_extract_selector(request))

    # Build prompt with system prompt if provided
    prompt = request.prompt
    if request.system_prompt:
        prompt = f"System: {request.system_prompt}\n\nUser: {prompt}\nAssistant:"

    if request.stream:
        return await generate_stream(request, prompt, bundle=bundle, model_key=model_key)

    # Non-streaming generation
    start_time = time.time()
    generated_text = ""
    token_count = 0

    async for token_text, should_stop in generate_tokens(
        prompt=prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        repetition_penalty=request.repetition_penalty,
        bundle=bundle,
        model_key=model_key,
        session_id=request.session_id,
        stop_sequences=request.stop_sequences,
    ):
        generated_text += token_text
        token_count += 1
        if should_stop:
            break

    elapsed = time.time() - start_time
    tok_per_sec = token_count / elapsed if elapsed > 0 else 0

    return GenerateResponse(
        text=generated_text,
        tokens_generated=token_count,
        tokens_per_second=tok_per_sec,
        session_id=request.session_id,
    )


async def generate_stream(
    request: GenerateRequest,
    prompt: str,
    *,
    bundle: Optional[ModelBundle] = None,
    model_key: Optional[str] = None,
):
    """Streaming generation with SSE"""
    if state.manager is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if bundle is None or model_key is None:
        bundle, model_key = await state.manager.get_bundle(_extract_selector(request))

    async def event_generator():
        start_time = time.time()
        token_count = 0

        async for token_text, should_stop in generate_tokens(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            bundle=bundle,
            model_key=model_key,
            session_id=request.session_id,
            stop_sequences=request.stop_sequences,
        ):
            token_count += 1
            data = json.dumps({"token": token_text, "done": should_stop})
            yield f"data: {data}\n\n"

            if should_stop:
                break

        elapsed = time.time() - start_time
        tok_per_sec = token_count / elapsed if elapsed > 0 else 0

        # Final message with stats
        final = json.dumps({
            "done": True,
            "tokens_generated": token_count,
            "tokens_per_second": tok_per_sec,
        })
        yield f"data: {final}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.post("/chat")
async def chat(request: ChatRequest):
    """Chat completion endpoint"""
    if state.manager is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    bundle, model_key = await state.manager.get_bundle(_extract_selector(request))

    # Build prompt from messages
    prompt = ""
    if request.system_prompt:
        prompt += f"System: {request.system_prompt}\n\n"

    for msg in request.messages:
        if msg.role == "user":
            prompt += f"User: {msg.content}\n"
        elif msg.role == "assistant":
            prompt += f"Assistant: {msg.content}\n"

    prompt += "Assistant:"

    if request.stream:
        return await generate_stream(
            GenerateRequest(
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,
                stream=True,
                session_id=request.session_id,
                model_id=request.model_id,
                artifact_uri=request.artifact_uri,
                checkpoint_path=request.checkpoint_path,
                model_config_path=request.model_config_path,
                eval=request.eval,
            ),
            prompt,
            bundle=bundle,
            model_key=model_key,
        )

    # Non-streaming
    start_time = time.time()
    generated_text = ""
    token_count = 0

    async for token_text, should_stop in generate_tokens(
        prompt=prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        repetition_penalty=request.repetition_penalty,
        bundle=bundle,
        model_key=model_key,
        session_id=request.session_id,
    ):
        generated_text += token_text
        token_count += 1
        if should_stop:
            break

    elapsed = time.time() - start_time
    tok_per_sec = token_count / elapsed if elapsed > 0 else 0

    return {
        "message": {"role": "assistant", "content": generated_text.strip()},
        "tokens_generated": token_count,
        "tokens_per_second": tok_per_sec,
        "session_id": request.session_id,
    }


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session's memory state"""
    deleted = 0
    for mk in list(state.sessions.keys()):
        if session_id in state.sessions.get(mk, {}):
            del state.sessions[mk][session_id]
            deleted += 1
    if deleted:
        return {"status": "deleted", "session_id": session_id, "deleted_count": deleted}
    raise HTTPException(status_code=404, detail="Session not found")


@app.get("/sessions")
async def list_sessions():
    """List active sessions"""
    out: Dict[str, Any] = {}
    for mk, sess in state.sessions.items():
        out[mk] = list(sess.keys())
    return {"sessions": out}


@app.get("/models_loaded")
async def models_loaded():
    """List cached model bundles (LRU order)."""
    if state.manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    return {"models": await state.manager.list_cached()}


@app.post("/cache/clear")
async def clear_cache():
    """Clear cached models and session memories."""
    if state.manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    n = await state.manager.clear_cache()
    state.sessions.clear()
    state.model = None
    state.config = None
    return {"cleared_models": n, "cleared_sessions": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
