use anyhow::Result;
use clam_core::{
    Interaction, Post, UserPreferenceCalculator, persistent_calc::PersistentPreferenceCalculator,
};
use embed_anything::embeddings::embed::Embedder;
use embed_anything::embeddings::local::model2vec::Model2VecEmbedder;
use sqlx::postgres::PgPoolOptions;
use std::time::Duration;
use uuid::Uuid;

#[tokio::test]
async fn test_preference_change_over_time() -> Result<()> {
    // 1. Setup Postgres Database Connection for posts (Items)
    let database_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://postgres:postgres@localhost:5432/postgres".to_string());

    let pg_pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&database_url)
        .await?;

    sqlx::query("CREATE EXTENSION IF NOT EXISTS vector")
        .execute(&pg_pool)
        .await?;
    sqlx::query("DROP TABLE IF EXISTS posts_persist")
        .execute(&pg_pool)
        .await?;

    let model_path = std::env::var("MODEL_PATH").expect("MODEL_PATH env var is not set");
    let embed_dim = 256;

    sqlx::query(&format!(
        "
        CREATE TABLE posts_persist (
            id UUID PRIMARY KEY,
            title TEXT NOT NULL,
            body TEXT NOT NULL,
            embedding vector({})
        )
    ",
        embed_dim
    ))
    .execute(&pg_pool)
    .await?;

    // 2. Setup Embedders and Calcs
    let simple_calc = UserPreferenceCalculator::new(&model_path)?; // For pre-filling

    let m2v = Model2VecEmbedder::new(&model_path, None, None)?;
    let embedder = Embedder::Text(embed_anything::embeddings::embed::TextEmbedder::Model2Vec(
        m2v.into(),
    ));

    // Persistent Calculator: half life of 1 second for fast test simulation
    let half_life_secs = 1.0;
    let persist_calc =
        PersistentPreferenceCalculator::new("sqlite::memory:", embedder, half_life_secs).await?;

    // 3. Simple Mock Posts Collection
    let mut posts = Vec::new();

    let sample_data = vec![
        (
            "Sports",
            "A great soccer match yesterday with lots of goals",
        ),
        ("Sports", "Tennis player wins championship tournament"),
        ("Tech", "New smartphone release brings better battery life"),
        ("Tech", "Artificial intelligence taking over mundane tasks"),
        ("Food", "How to bake a perfect sourdough bread at home"),
        ("Food", "Delicious pasta recipe using fresh tomatoes"),
    ];

    for (category, text) in sample_data {
        let post = Post {
            id: Uuid::new_v4().to_string(),
            title: category.to_string(),
            body: text.to_string(),
        };

        // Pre-calculate embedding for indexing
        let interactions = vec![(post.clone(), Interaction::Liked)];
        let embedding = simple_calc
            .calculate_preference_vector(&interactions)
            .await?
            .unwrap();

        sqlx::query(
            "INSERT INTO posts_persist (id, title, body, embedding) VALUES ($1, $2, $3, $4::vector)",
        )
        .bind(Uuid::parse_str(&post.id)?)
        .bind(&post.title)
        .bind(&post.body)
        .bind(embedding)
        .execute(&pg_pool)
        .await?;

        posts.push(post);
    }

    // Identify some posts for interactions
    let sports_post = posts.iter().find(|p| p.title == "Sports").unwrap();
    let food_post = posts.iter().find(|p| p.title == "Food").unwrap();

    let test_user = "user_time_test";

    // 4. Initial phase: User interacts with Sports
    persist_calc
        .add_interaction(test_user, sports_post, Interaction::Liked)
        .await?;
    persist_calc
        .add_interaction(test_user, sports_post, Interaction::Liked)
        .await?;

    // Retrieval right after Sports
    let pref1 = persist_calc.get_user_preference(test_user).await?.unwrap();

    #[derive(sqlx::FromRow)]
    struct SearchResult {
        title: String,
        distance: f64,
    }

    let results1 = sqlx::query_as::<_, SearchResult>(
        "SELECT title, (embedding <=> $1::vector)::float8 as distance FROM posts_persist ORDER BY distance ASC LIMIT 2"
    )
    .bind(pref1)
    .fetch_all(&pg_pool)
    .await?;

    println!("--- Phase 1: User Likes Sports ---");
    for res in &results1 {
        println!("{} (distance: {:.4})", res.title, res.distance);
    }
    assert_eq!(results1.first().unwrap().title, "Sports"); // Should prioritize Sports

    // 5. Wait for decay to happen (2 seconds = 2 half lives -> weight drops to 25%)
    println!("Waiting 2 seconds for decay...");
    tokio::time::sleep(Duration::from_secs(2)).await;

    // 6. Next phase: User's interest shifts completely to Food
    persist_calc
        .add_interaction(test_user, food_post, Interaction::Liked)
        .await?;
    persist_calc
        .add_interaction(test_user, food_post, Interaction::Shared)
        .await?;

    // New Retrieval
    let pref2 = persist_calc.get_user_preference(test_user).await?.unwrap();
    let results2 = sqlx::query_as::<_, SearchResult>(
        "SELECT title, (embedding <=> $1::vector)::float8 as distance FROM posts_persist ORDER BY distance ASC LIMIT 2"
    )
    .bind(pref2)
    .fetch_all(&pg_pool)
    .await?;

    println!("--- Phase 2: User shifts to Food (after Sports decayed) ---");
    for res in &results2 {
        println!("{} (distance: {:.4})", res.title, res.distance);
    }

    // Because of the exponential decay, Food should heavily dominate now.
    assert_eq!(results2.first().unwrap().title, "Food");

    Ok(())
}
