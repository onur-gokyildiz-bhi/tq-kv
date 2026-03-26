use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use tiny_http::{Header, Method, Response, Server};

use crate::{chat, engine::{Engine, GenerationParams}, inference};

#[derive(Deserialize, Serialize)]
pub struct InferRequest {
    pub system: String,
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default = "default_repeat_penalty")]
    pub repeat_penalty: f32,
}

fn default_max_tokens() -> u32 { 1024 }
fn default_temperature() -> f32 { 0.7 }
fn default_top_p() -> f32 { 0.9 }
fn default_top_k() -> usize { 40 }
fn default_repeat_penalty() -> f32 { 1.1 }

#[derive(Serialize, Deserialize)]
pub struct InferResponse {
    pub text: String,
}

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub engine: String,
}

#[derive(Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

/// HTTP daemon — model loaded once, requests serialized via Mutex
pub fn run_daemon(engine: Engine, template: &chat::ChatTemplate, port: u16) -> Result<()> {
    let template = template.clone();
    let engine_mutex = Mutex::new(engine);

    let bind_addr = format!("0.0.0.0:{}", port);
    let server = Server::http(&bind_addr)
        .map_err(|e| anyhow::anyhow!("HTTP server failed: {}", e))?;

    eprintln!("========================================");
    eprintln!("  TurboQuant Engine Daemon");
    eprintln!("  http://localhost:{}", port);
    eprintln!("  Ctrl+C to stop");
    eprintln!("========================================");

    for mut request in server.incoming_requests() {
        let url = request.url().to_string();
        let method = request.method().clone();

        match (method, url.as_str()) {
            (Method::Get, "/health") => {
                let resp = HealthResponse { status: "ok".into(), engine: "tq-kv".into() };
                let json = serde_json::to_string(&resp).unwrap();
                let _ = request.respond(Response::from_string(json).with_header(json_header()));
            }

            (Method::Post, "/infer") => {
                let mut raw_body = Vec::new();
                if let Err(e) = std::io::Read::read_to_end(&mut request.as_reader(), &mut raw_body) {
                    respond_error(request, 400, &format!("Body read error: {}", e));
                    continue;
                }
                let body = String::from_utf8_lossy(&raw_body).into_owned();

                let infer_req: InferRequest = match serde_json::from_str(&body) {
                    Ok(r) => r,
                    Err(e) => {
                        respond_error(request, 400, &format!("JSON error: {}", e));
                        continue;
                    }
                };

                let params = GenerationParams {
                    max_tokens: infer_req.max_tokens,
                    temperature: infer_req.temperature,
                    top_p: infer_req.top_p,
                    top_k: infer_req.top_k,
                    repeat_penalty: infer_req.repeat_penalty,
                    ..Default::default()
                };

                let formatted = chat::format_chat(&template, &infer_req.system, &infer_req.prompt);

                let result = {
                    let mut engine = engine_mutex.lock().unwrap();
                    engine.clear_cache();
                    inference::generate_silent(&mut engine, &formatted, &params)
                };

                match result {
                    Ok(text) => {
                        let resp = InferResponse { text };
                        let json = serde_json::to_string(&resp).unwrap();
                        let _ = request.respond(Response::from_string(json).with_header(json_header()));
                    }
                    Err(e) => {
                        respond_error(request, 500, &e.to_string());
                    }
                }
            }

            _ => {
                respond_error(request, 404, "Unknown endpoint");
            }
        }
    }

    Ok(())
}

fn json_header() -> Header {
    Header::from_bytes("Content-Type", "application/json; charset=utf-8").unwrap()
}

fn respond_error(request: tiny_http::Request, status: u16, msg: &str) {
    let err = ErrorResponse { error: msg.to_string() };
    let json = serde_json::to_string(&err).unwrap();
    let _ = request.respond(
        Response::from_string(json).with_status_code(status).with_header(json_header()),
    );
}
