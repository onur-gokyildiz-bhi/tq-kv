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
use crate::engine::{Engine, GenerationParams};

// ---------------------------------------------------------------------------
// Application state
// ---------------------------------------------------------------------------

struct AppState {
    engine: Mutex<Engine>,
    template: ChatTemplate,
    model_name: String,
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
) -> Result<()> {
    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
        template,
        model_name,
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
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
    eprintln!("    POST /v1/chat/completions  (OpenAI compatible)");
    eprintln!("    GET  /v1/models");
    eprintln!("    GET  /health");
    eprintln!("  Ctrl+C to stop");
    eprintln!("========================================");

    axum::serve(listener, app).await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Route handlers
// ---------------------------------------------------------------------------

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".into(),
        engine: "tq-kv".into(),
    })
}

async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelListResponse> {
    Json(ModelListResponse {
        object: "list".into(),
        data: vec![ModelEntry {
            id: state.model_name.clone(),
            object: "model".into(),
            owned_by: "local".into(),
        }],
    })
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> std::result::Result<impl IntoResponse, (StatusCode, Json<ErrorBody>)> {
    let stream = req.stream.unwrap_or(false);

    // Build prompt from messages
    let (system_prompt, user_prompt) = extract_prompts(&req.messages);
    let template = state.template.clone();
    let formatted = chat::format_chat(&template, &system_prompt, &user_prompt);

    let params = GenerationParams {
        max_tokens: req.max_tokens.unwrap_or(512),
        temperature: req.temperature.unwrap_or(0.7),
        top_p: req.top_p.unwrap_or(0.9),
        ..Default::default()
    };

    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let model_name = state.model_name.clone();
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
            let mut engine = state.engine.lock().unwrap();
            engine.clear_cache();
            let _ = engine.generate(&formatted, &params, |token| {
                let _ = tx.blocking_send(token.to_string());
            });
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
            let mut engine = state_clone.engine.lock().unwrap();
            engine.clear_cache();
            engine.generate_silent(&formatted, &params)
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
    let template = state.template.clone();
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
        let mut engine = state.engine.lock().unwrap();
        engine.clear_cache();
        engine.generate_silent(&formatted, &params)
    })
    .await
    .map_err(|e| api_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()))?
    .map_err(|e| api_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()))?;

    Ok(Json(LegacyInferResponse { text: result }))
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
