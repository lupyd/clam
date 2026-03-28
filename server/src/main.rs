use anyhow::Result;
use deadpool_postgres::{Config, ManagerConfig, Pool, RecyclingMethod, Runtime};
use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Method, Request, Response, Server, StatusCode};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::net::SocketAddr;
use tokio_postgres::NoTls;

#[derive(Serialize, Deserialize, Debug)]
struct EmbeddingPayload {
    id: uuid::Uuid,
    embedding: Vec<f32>,
}

async fn handle_request(req: Request<Body>, pool: Pool) -> Result<Response<Body>, hyper::Error> {
    match (req.method(), req.uri().path()) {
        (&Method::POST, "/embeddings") => {
            let body_bytes = hyper::body::to_bytes(req.into_body()).await?;
            let payload: EmbeddingPayload = match serde_json::from_slice(&body_bytes) {
                Ok(p) => p,
                Err(_) => {
                    let mut bad_request = Response::default();
                    *bad_request.status_mut() = StatusCode::BAD_REQUEST;
                    return Ok(bad_request);
                }
            };

            let client = match pool.get().await {
                Ok(c) => c,
                Err(_) => {
                    let mut error = Response::default();
                    *error.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
                    return Ok(error);
                }
            };

            let stmt = match client
                .prepare("INSERT INTO embeddings (id, vector) VALUES ($1, $2) ON CONFLICT (id) DO UPDATE SET vector = EXCLUDED.vector")
                .await
            {
                Ok(s) => s,
                Err(_) => {
                    let mut error = Response::default();
                    *error.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
                    return Ok(error);
                }
            };

            if let Err(_) = client.execute(&stmt, &[&payload.id, &payload.embedding]).await {
                let mut error = Response::default();
                *error.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
                return Ok(error);
            }

            Ok(Response::new(Body::from("Saved")))
        }
        (&Method::GET, path) if path.starts_with("/embeddings/") => {
            let id_str = path.trim_start_matches("/embeddings/");
            let id = match uuid::Uuid::parse_str(id_str) {
                Ok(i) => i,
                Err(_) => {
                    let mut bad_request = Response::default();
                    *bad_request.status_mut() = StatusCode::BAD_REQUEST;
                    return Ok(bad_request);
                }
            };

            let client = match pool.get().await {
                Ok(c) => c,
                Err(_) => {
                    let mut error = Response::default();
                    *error.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
                    return Ok(error);
                }
            };

            let stmt = match client.prepare("SELECT vector FROM embeddings WHERE id = $1").await {
                Ok(s) => s,
                Err(_) => {
                    let mut error = Response::default();
                    *error.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
                    return Ok(error);
                }
            };

            match client.query_opt(&stmt, &[&id]).await {
                Ok(Some(row)) => {
                    let embedding: Vec<f32> = row.get(0);
                    let payload = EmbeddingPayload { id, embedding };
                    let json = serde_json::to_string(&payload).unwrap();
                    let mut response = Response::new(Body::from(json));
                    response.headers_mut().insert("Content-Type", "application/json".parse().unwrap());
                    Ok(response)
                }
                Ok(None) => {
                    let mut not_found = Response::default();
                    *not_found.status_mut() = StatusCode::NOT_FOUND;
                    Ok(not_found)
                }
                Err(_) => {
                    let mut error = Response::default();
                    *error.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
                    Ok(error)
                }
            }
        }
        _ => {
            let mut not_found = Response::default();
            *not_found.status_mut() = StatusCode::NOT_FOUND;
            Ok(not_found)
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut cfg = Config::new();
    // Default config can be tailored via environment variables usually, but we fallback
    // to localhost and postgres as default for this example.
    cfg.host = Some(std::env::var("PG_HOST").unwrap_or_else(|_| "localhost".to_string()));
    cfg.user = Some(std::env::var("PG_USER").unwrap_or_else(|_| "postgres".to_string()));
    cfg.dbname = Some(std::env::var("PG_DBNAME").unwrap_or_else(|_| "postgres".to_string()));
    cfg.manager = Some(ManagerConfig {
        recycling_method: RecyclingMethod::Fast,
    });

    let pool = cfg.create_pool(Some(Runtime::Tokio1), NoTls)?;

    // Make sure table exists
    {
        let client = pool.get().await?;
        client
            .execute(
                "CREATE TABLE IF NOT EXISTS embeddings (
                    id UUID PRIMARY KEY,
                    vector REAL[]
                )",
                &[],
            )
            .await?;
    }

    let make_svc = make_service_fn(move |_conn| {
        let pool = pool.clone();
        async move {
            Ok::<_, Infallible>(service_fn(move |req| {
                let pool = pool.clone();
                handle_request(req, pool)
            }))
        }
    });

    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    let server = Server::bind(&addr).serve(make_svc);

    println!("Listening on http://{}", addr);

    if let Err(e) = server.await {
        eprintln!("server error: {}", e);
    }

    Ok(())
}
