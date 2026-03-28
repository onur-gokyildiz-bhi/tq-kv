//! Axum-based async HTTP server with OpenAI-compatible API.
//!
//! Model inference is sync (candle) so all model calls go through `spawn_blocking`.

use std::convert::Infallible;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tower_http::cors::CorsLayer;

use crate::chat::{self, ChatTemplate};
use crate::config;
use crate::engine::{Engine, GenerationParams};
use crate::hub;

// ---------------------------------------------------------------------------
// Application state
// ---------------------------------------------------------------------------

struct AppState {
    engine: Mutex<Option<Engine>>,
    template: Mutex<ChatTemplate>,
    model_name: Mutex<String>,
    tq_config: Option<tq_kv::TurboQuantConfig>,
    force_cpu: bool,
}

// ---------------------------------------------------------------------------
// OpenAI-compatible request/response types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
#[allow(dead_code)]
struct ChatCompletionRequest {
    #[serde(default)]
    model: String,
    messages: Vec<Message>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default = "default_temperature")]
    temperature: Option<f32>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    stop: Option<Vec<String>>,
}

#[derive(Deserialize, Serialize, Clone)]
struct Message {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<Choice>,
    usage: Usage,
}

#[derive(Serialize)]
struct Choice {
    index: u32,
    message: Message,
    finish_reason: String,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[derive(Serialize)]
struct StreamChoice {
    index: u32,
    delta: DeltaMessage,
    finish_reason: Option<String>,
}

#[derive(Serialize)]
struct DeltaMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

#[derive(Serialize)]
struct StreamChunk {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<StreamChoice>,
}

#[derive(Serialize)]
struct ModelListResponse {
    object: String,
    data: Vec<ModelEntry>,
}

#[derive(Serialize)]
struct ModelEntry {
    id: String,
    object: String,
    owned_by: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    active: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    size_gb: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    arch: Option<String>,
}

#[derive(Deserialize)]
struct LoadModelRequest {
    model: String,
}

#[derive(Serialize)]
struct LoadModelResponse {
    status: String,
    model: String,
    message: String,
}

#[derive(Serialize)]
struct ModelStatusResponse {
    model: String,
    status: String,
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    engine: String,
}

#[derive(Serialize)]
struct ErrorBody {
    error: ErrorDetail,
}

#[derive(Serialize)]
struct ErrorDetail {
    message: String,
    r#type: String,
}

fn default_temperature() -> Option<f32> {
    Some(0.7)
}

// ---------------------------------------------------------------------------
// Server entry point
// ---------------------------------------------------------------------------

pub async fn run_server(
    engine: Engine,
    template: ChatTemplate,
    model_name: String,
    port: u16,
    tq_config: Option<tq_kv::TurboQuantConfig>,
    force_cpu: bool,
) -> Result<()> {
    let state = Arc::new(AppState {
        engine: Mutex::new(Some(engine)),
        template: Mutex::new(template),
        model_name: Mutex::new(model_name),
        tq_config,
        force_cpu,
    });

    let app = Router::new()
        .route("/", get(serve_ui))
        .route("/index.html", get(serve_ui))
        .route("/health", get(health))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .route("/v1/models/load", post(load_model))
        .route("/v1/models/status", get(model_status))
        // Legacy endpoint for backward compat
        .route("/infer", post(legacy_infer))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let bind_addr = format!("0.0.0.0:{}", port);
    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;

    eprintln!("========================================");
    eprintln!("  TurboQuant Engine Server (axum)");
    eprintln!("  http://localhost:{}", port);
    eprintln!("  Endpoints:");
    eprintln!("    GET  /                     (Web UI)");
    eprintln!("    POST /v1/chat/completions  (OpenAI compatible)");
    eprintln!("    GET  /v1/models");
    eprintln!("    POST /v1/models/load");
    eprintln!("    GET  /v1/models/status");
    eprintln!("    GET  /health");
    eprintln!("  Ctrl+C to stop");
    eprintln!("========================================");

    axum::serve(listener, app).await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Route handlers
// ---------------------------------------------------------------------------

async fn serve_ui() -> axum::response::Html<&'static str> {
    axum::response::Html(include_str!("web/index.html"))
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".into(),
        engine: "tq-kv".into(),
    })
}

async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelListResponse> {
    let current_model = state.model_name.lock().unwrap().clone();
    let downloaded = hub::list_downloaded();

    let mut models: Vec<ModelEntry> = downloaded
        .iter()
        .filter(|dm| dm.gguf_exists)
        .map(|dm| {
            let id = format!("{}:{}", dm.meta.name, dm.meta.tag);
            let is_active = id == current_model || dm.meta.display == current_model;
            ModelEntry {
                id,
                object: "model".into(),
                owned_by: "local".into(),
                active: Some(is_active),
                size_gb: Some(dm.meta.size_gb),
                arch: Some(dm.meta.arch.clone()),
            }
        })
        .collect();

    // If no downloaded models matched the current one (e.g. loaded via path),
    // still include the currently loaded model at the top.
    if !models.iter().any(|m| m.active == Some(true)) {
        models.insert(
            0,
            ModelEntry {
                id: current_model,
                object: "model".into(),
                owned_by: "local".into(),
                active: Some(true),
                size_gb: None,
                arch: None,
            },
        );
    }

    Json(ModelListResponse {
        object: "list".into(),
        data: models,
    })
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> std::result::Result<impl IntoResponse, (StatusCode, Json<ErrorBody>)> {
    // Check if engine is loaded
    {
        let engine_guard = state.engine.lock().unwrap();
        if engine_guard.is_none() {
            return Err(api_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "Model is loading, please wait...",
            ));
        }
    }

    let stream = req.stream.unwrap_or(false);

    // Build prompt from messages
    let (system_prompt, user_prompt) = extract_prompts(&req.messages);
    let template = state.template.lock().unwrap().clone();
    let formatted = chat::format_chat(&template, &system_prompt, &user_prompt);

    let params = GenerationParams {
        max_tokens: req.max_tokens.unwrap_or(512),
        temperature: req.temperature.unwrap_or(0.7),
        top_p: req.top_p.unwrap_or(0.9),
        ..Default::default()
    };

    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let model_name = state.model_name.lock().unwrap().clone();
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    if stream {
        // SSE streaming path
        let (tx, rx) = tokio::sync::mpsc::channel::<String>(32);
        let req_id = request_id.clone();
        let model = model_name.clone();

        tokio::task::spawn_blocking(move || {
            let mut guard = state.engine.lock().unwrap();
            if let Some(ref mut engine) = *guard {
                engine.clear_cache();
                let _ = engine.generate(&formatted, &params, |token| {
                    let _ = tx.blocking_send(token.to_string());
                });
            }
            // Signal end
            drop(tx);
        });

        let stream = ReceiverStream::new(rx).map(move |token| -> std::result::Result<Event, Infallible> {
            let chunk = StreamChunk {
                id: req_id.clone(),
                object: "chat.completion.chunk".into(),
                created: now,
                model: model.clone(),
                choices: vec![StreamChoice {
                    index: 0,
                    delta: DeltaMessage {
                        role: None,
                        content: Some(token),
                    },
                    finish_reason: None,
                }],
            };
            let json = serde_json::to_string(&chunk).unwrap_or_default();
            Ok(Event::default().data(json))
        });

        Ok(Sse::new(stream).into_response())
    } else {
        // Non-streaming path
        let state_clone = state.clone();
        let result = tokio::task::spawn_blocking(move || {
            let mut guard = state_clone.engine.lock().unwrap();
            match guard.as_mut() {
                Some(engine) => {
                    engine.clear_cache();
                    engine.generate_silent(&formatted, &params)
                }
                None => Err(anyhow::anyhow!("Model is loading, please wait...")),
            }
        })
        .await
        .map_err(|e| api_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()))?
        .map_err(|e| api_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()))?;

        let response = ChatCompletionResponse {
            id: request_id,
            object: "chat.completion".into(),
            created: now,
            model: model_name,
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: "assistant".into(),
                    content: result.clone(),
                },
                finish_reason: "stop".into(),
            }],
            usage: Usage {
                prompt_tokens: 0, // TODO: count actual tokens
                completion_tokens: 0,
                total_tokens: 0,
            },
        };

        Ok(Json(response).into_response())
    }
}

// Legacy /infer endpoint for backward compat with old clients

#[derive(Deserialize)]
struct LegacyInferRequest {
    system: String,
    prompt: String,
    #[serde(default = "default_legacy_max_tokens")]
    max_tokens: u32,
    #[serde(default = "default_legacy_temperature")]
    temperature: f32,
    #[serde(default = "default_legacy_top_p")]
    top_p: f32,
    #[serde(default = "default_legacy_top_k")]
    top_k: usize,
    #[serde(default = "default_legacy_repeat_penalty")]
    repeat_penalty: f32,
}

fn default_legacy_max_tokens() -> u32 { 1024 }
fn default_legacy_temperature() -> f32 { 0.7 }
fn default_legacy_top_p() -> f32 { 0.9 }
fn default_legacy_top_k() -> usize { 40 }
fn default_legacy_repeat_penalty() -> f32 { 1.1 }

#[derive(Serialize)]
struct LegacyInferResponse {
    text: String,
}

async fn legacy_infer(
    State(state): State<Arc<AppState>>,
    Json(req): Json<LegacyInferRequest>,
) -> std::result::Result<Json<LegacyInferResponse>, (StatusCode, Json<ErrorBody>)> {
    let template = state.template.lock().unwrap().clone();
    let formatted = chat::format_chat(&template, &req.system, &req.prompt);

    let params = GenerationParams {
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        repeat_penalty: req.repeat_penalty,
        ..Default::default()
    };

    let result = tokio::task::spawn_blocking(move || {
        let mut guard = state.engine.lock().unwrap();
        match guard.as_mut() {
            Some(engine) => {
                engine.clear_cache();
                engine.generate_silent(&formatted, &params)
            }
            None => Err(anyhow::anyhow!("Model is loading, please wait...")),
        }
    })
    .await
    .map_err(|e| api_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()))?
    .map_err(|e| api_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()))?;

    Ok(Json(LegacyInferResponse { text: result }))
}

async fn load_model(
    State(state): State<Arc<AppState>>,
    Json(req): Json<LoadModelRequest>,
) -> std::result::Result<Json<LoadModelResponse>, (StatusCode, Json<ErrorBody>)> {
    let model_query = req.model.clone();
    let tq_config = state.tq_config.clone();
    let force_cpu = state.force_cpu;

    // Set engine to None while loading (signals "loading" to other endpoints)
    {
        let mut guard = state.engine.lock().unwrap();
        *guard = None;
    }

    let state_clone = state.clone();
    let query = model_query.clone();

    let result = tokio::task::spawn_blocking(move || -> anyhow::Result<(Engine, String, String)> {
        let (gguf_path, tokenizer_path) = hub::resolve(&query)?;
        let tok_path = tokenizer_path.ok_or_else(|| {
            anyhow::anyhow!("No tokenizer found for model '{}'. Pull the model first with `tq pull {}`.", query, query)
        })?;

        let mf = gguf_path.to_string_lossy().to_string();
        let arch = config::detect_arch(&mf);

        let display = crate::catalog::find(&query)
            .map(|e| e.display.to_string())
            .unwrap_or_else(|| query.clone());

        eprintln!("Loading model: {} ({})", display, query);
        let engine = Engine::load_with_device(
            &gguf_path, &tok_path, arch, tq_config, force_cpu,
        )?;
        eprintln!("Model loaded: {}", display);

        Ok((engine, mf, display))
    })
    .await
    .map_err(|e| api_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()))?;

    match result {
        Ok((engine, model_file, display_name)) => {
            let template = ChatTemplate::detect(&model_file);
            {
                let mut guard = state_clone.engine.lock().unwrap();
                *guard = Some(engine);
            }
            {
                let mut t = state_clone.template.lock().unwrap();
                *t = template;
            }
            {
                let mut n = state_clone.model_name.lock().unwrap();
                *n = display_name.clone();
            }

            Ok(Json(LoadModelResponse {
                status: "ok".into(),
                model: display_name,
                message: format!("Model '{}' loaded successfully.", model_query),
            }))
        }
        Err(e) => {
            // Loading failed — engine stays None. The user should retry or
            // load a different model. We leave model_name unchanged so
            // the UI can show what was previously loaded.
            Err(api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("Failed to load model '{}': {}", model_query, e),
            ))
        }
    }
}

async fn model_status(
    State(state): State<Arc<AppState>>,
) -> Json<ModelStatusResponse> {
    let model_name = state.model_name.lock().unwrap().clone();
    let engine_guard = state.engine.lock().unwrap();
    let status = if engine_guard.is_some() {
        "ready"
    } else {
        "loading"
    };

    Json(ModelStatusResponse {
        model: model_name,
        status: status.into(),
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn extract_prompts(messages: &[Message]) -> (String, String) {
    let mut system = String::new();
    let mut user_parts = Vec::new();

    for msg in messages {
        match msg.role.as_str() {
            "system" => system = msg.content.clone(),
            "user" => user_parts.push(msg.content.clone()),
            "assistant" => {} // skip for now in single-turn
            _ => {}
        }
    }

    if system.is_empty() {
        system = "You are a helpful assistant.".into();
    }

    let user = user_parts.join("\n");
    (system, user)
}

fn api_error(status: StatusCode, message: &str) -> (StatusCode, Json<ErrorBody>) {
    (
        status,
        Json(ErrorBody {
            error: ErrorDetail {
                message: message.to_string(),
                r#type: "server_error".into(),
            },
        }),
    )
}
