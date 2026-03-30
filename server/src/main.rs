use anyhow::Result;
use deadpool_postgres::{Config, ManagerConfig, Pool, RecyclingMethod, Runtime};
use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Method, Request, Response, Server, StatusCode};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::net::SocketAddr;
use tokio_postgres::types::{ToSql, Type};
use tokio_postgres::NoTls;

#[derive(Serialize, Deserialize, Debug)]
struct EmbeddingPayload {
    id: uuid::Uuid,
    embedding: Vec<f32>,
}

#[derive(Serialize, Deserialize, Debug)]
struct SearchPayload {
    vector: Vec<f32>,
    #[serde(default = "default_limit")]
    limit: i64,
}

fn default_limit() -> i64 {
    5
}

#[derive(Serialize, Deserialize, Debug)]
struct SearchResult {
    id: uuid::Uuid,
    distance: f64,
}

async fn handle_request_inner(req: Request<Body>, pool: Pool) -> Result<Response<Body>, hyper::Error> {
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

            let id_param: &(dyn ToSql + Sync) = &payload.id;
            let vec_param: &(dyn ToSql + Sync) = &payload.embedding;
            let params: &[(&(dyn ToSql + Sync), Type)] =
                &[(id_param, Type::UUID), (vec_param, Type::FLOAT4_ARRAY)];

            if let Err(e) = client.query_typed(
                "INSERT INTO embeddings (id, vector) VALUES ($1, $2) ON CONFLICT (id) DO UPDATE SET vector = EXCLUDED.vector",
                params
            ).await {
                eprintln!("Insert error: {:?}", e);
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

            let id_param: &(dyn ToSql + Sync) = &id;
            let params: &[(&(dyn ToSql + Sync), Type)] = &[(id_param, Type::UUID)];

            match client
                .query_typed("SELECT vector FROM embeddings WHERE id = $1", params)
                .await
            {
                Ok(rows) => {
                    if let Some(row) = rows.first() {
                        let embedding: Vec<f32> = row.get(0);
                        let payload = EmbeddingPayload { id, embedding };
                        let json = serde_json::to_string(&payload).unwrap();
                        let mut response = Response::new(Body::from(json));
                        response
                            .headers_mut()
                            .insert("Content-Type", "application/json".parse().unwrap());
                        Ok(response)
                    } else {
                        let mut not_found = Response::default();
                        *not_found.status_mut() = StatusCode::NOT_FOUND;
                        Ok(not_found)
                    }
                }
                Err(e) => {
                    eprintln!("Query error: {:?}", e);
                    let mut error = Response::default();
                    *error.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
                    Ok(error)
                }
            }
        }
        (&Method::POST, "/embeddings/closest") => {
            let body_bytes = hyper::body::to_bytes(req.into_body()).await?;
            let payload: SearchPayload = match serde_json::from_slice(&body_bytes) {
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

            // Using pgvector's cosine distance operator <=>
            // Casting REAL[] to vector allows us to use vector operators
            let vector_param: &(dyn ToSql + Sync) = &payload.vector;
            let limit_param: &(dyn ToSql + Sync) = &payload.limit;
            let params: &[(&(dyn ToSql + Sync), Type)] = &[
                (vector_param, Type::FLOAT4_ARRAY),
                (limit_param, Type::INT8),
            ];

            let rows = match client.query_typed(
                "SELECT id, (vector::vector <=> $1::vector)::float8 as distance FROM embeddings ORDER BY distance ASC LIMIT $2",
                params
            ).await {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("Query error: {:?}", e);
                    let mut error = Response::default();
                    *error.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
                    return Ok(error);
                }
            };

            let results: Vec<SearchResult> = rows
                .into_iter()
                .map(|row| SearchResult {
                    id: row.get(0),
                    distance: row.get(1),
                })
                .collect();

            let json = serde_json::to_string(&results).unwrap();
            let mut response = Response::new(Body::from(json));
            response
                .headers_mut()
                .insert("Content-Type", "application/json".parse().unwrap());
            Ok(response)
        }
        _ => {
            let mut not_found = Response::default();
            *not_found.status_mut() = StatusCode::NOT_FOUND;
            Ok(not_found)
        }
    }
}

async fn handle_request(req: Request<Body>, pool: Pool) -> Result<Response<Body>, hyper::Error> {
    if req.method() == Method::OPTIONS {
        let mut response = Response::default();
        response.headers_mut().insert("Access-Control-Allow-Origin", "*".parse().unwrap());
        response.headers_mut().insert("Access-Control-Allow-Methods", "POST, GET, OPTIONS".parse().unwrap());
        response.headers_mut().insert("Access-Control-Allow-Headers", "Content-Type".parse().unwrap());
        return Ok(response);
    }

    let mut response = handle_request_inner(req, pool).await?;
    response.headers_mut().insert("Access-Control-Allow-Origin", "*".parse().unwrap());
    Ok(response)
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

    // Make sure extensions and table exist
    {
        let client = pool.get().await?;
        client
            .batch_execute(
                "CREATE EXTENSION IF NOT EXISTS vector;
                 CREATE TABLE IF NOT EXISTS embeddings (
                    id UUID PRIMARY KEY,
                    vector REAL[]
                )",
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
