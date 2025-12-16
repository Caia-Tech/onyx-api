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
from pathlib import Path
from typing import Optional, List, Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add model path to sys.path
MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "onyx"
sys.path.insert(0, str(MODEL_DIR))

from onyx_model import Onyx, OnyxConfig

try:
    from transformers import AutoTokenizer
except ImportError:
    raise RuntimeError("transformers not installed. Run: pip install transformers")


# =============================================================================
# Configuration
# =============================================================================

class ServerConfig:
    checkpoint_path: str = str(MODEL_DIR / "checkpoints" / "checkpoint_step_1000.pt")
    model_config_path: str = str(MODEL_DIR / "models" / "150m" / "config.json")
    tokenizer_name: str = "NousResearch/Hermes-2-Pro-Llama-3-8B"
    device: str = "auto"  # auto, cuda, mps, cpu
    dtype: str = "float32"  # float32, float16, bfloat16
    max_seq_len: int = 4096
    default_max_tokens: int = 512
    default_temperature: float = 0.8
    default_top_p: float = 0.9
    default_top_k: int = 50


config = ServerConfig()


# =============================================================================
# Global State
# =============================================================================

class ModelState:
    model: Optional[Onyx] = None
    tokenizer: Any = None
    device: torch.device = None
    dtype: torch.dtype = None
    config: Optional[OnyxConfig] = None
    # Session memory states keyed by session_id
    sessions: Dict[str, List[Dict[str, Any]]] = {}


state = ModelState()


# =============================================================================
# Model Loading
# =============================================================================

def load_model_config(config_path: str) -> OnyxConfig:
    """Load model config from JSON file"""
    import dataclasses

    with open(config_path) as f:
        cfg_json = json.load(f)

    flat_cfg = {}
    for key, value in cfg_json.items():
        if isinstance(value, dict):
            flat_cfg.update(value)
        else:
            flat_cfg[key] = value

    valid_fields = {f.name for f in dataclasses.fields(OnyxConfig)}
    filtered_cfg = {k: v for k, v in flat_cfg.items() if k in valid_fields}
    return OnyxConfig(**filtered_cfg)


def load_model():
    """Load model and tokenizer"""
    print(f"Loading model config from: {config.model_config_path}")
    model_config = load_model_config(config.model_config_path)
    state.config = model_config

    # Setup device
    if config.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(config.device)
    state.device = device

    # Setup dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    state.dtype = dtype_map[config.dtype]

    print(f"Device: {device} | Dtype: {state.dtype}")

    # Load tokenizer
    print(f"Loading tokenizer: {config.tokenizer_name}")
    state.tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name,
        trust_remote_code=True
    )

    # Load model
    print(f"Loading checkpoint: {config.checkpoint_path}")
    checkpoint = torch.load(config.checkpoint_path, map_location='cpu', weights_only=False)

    model = Onyx(model_config)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device=device, dtype=state.dtype)
    model.eval()
    state.model = model

    print(f"Model loaded: {model.get_num_params():,} parameters")


# =============================================================================
# Request/Response Models
# =============================================================================

class GenerateRequest(BaseModel):
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


class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int
    tokens_per_second: float
    session_id: Optional[str] = None


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)
    stream: bool = Field(default=False)
    session_id: Optional[str] = Field(default=None)
    system_prompt: Optional[str] = Field(default=None)


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
    generated_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sample next token"""

    if repetition_penalty != 1.0 and generated_tokens is not None and generated_tokens.numel() > 0:
        for token_id in generated_tokens.unique():
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
    session_id: Optional[str] = None,
    stop_sequences: Optional[List[str]] = None,
) -> AsyncGenerator[tuple[str, bool], None]:
    """Generate tokens, yielding each one"""

    model = state.model
    tokenizer = state.tokenizer
    device = state.device

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    B, S = input_ids.shape

    # Get or create memory states
    if session_id and session_id in state.sessions:
        memory_states = state.sessions[session_id]
    else:
        memory_states = model.init_memory_states(B, device, state.dtype)

    # Get stop token IDs
    stop_token_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id else []
    for token in ["<|eot_id|>", "<|end|>", "</s>", "<|im_end|>"]:
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

    generated_tokens = torch.tensor([], dtype=torch.long, device=device)
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
                generated_tokens=generated_tokens,
            )

            generated_tokens = torch.cat([generated_tokens, next_token.squeeze(0)])
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
        state.sessions[session_id] = memory_states


# =============================================================================
# API Endpoints
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup"""
    load_model()
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
        "model_loaded": state.model is not None,
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
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Build prompt with system prompt if provided
    prompt = request.prompt
    if request.system_prompt:
        prompt = f"System: {request.system_prompt}\n\nUser: {prompt}\nAssistant:"

    if request.stream:
        return await generate_stream(request, prompt)

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


async def generate_stream(request: GenerateRequest, prompt: str):
    """Streaming generation with SSE"""

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
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

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
            ),
            prompt
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
    if session_id in state.sessions:
        del state.sessions[session_id]
        return {"status": "deleted", "session_id": session_id}
    raise HTTPException(status_code=404, detail="Session not found")


@app.get("/sessions")
async def list_sessions():
    """List active sessions"""
    return {"sessions": list(state.sessions.keys())}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
