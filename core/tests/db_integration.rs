use anyhow::Result;
use clam_core::{Interaction, Post, UserPreferenceCalculator};
use sqlx::postgres::PgPoolOptions;
use uuid::Uuid;

#[tokio::test]
async fn test_database_integration() -> Result<()> {
    // 1. Setup Database Connection
    let database_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://postgres:postgres@localhost:5432/postgres".to_string());

    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&database_url)
        .await?;

    sqlx::query("CREATE EXTENSION IF NOT EXISTS vector")
        .execute(&pool)
        .await?;
    sqlx::query("DROP TABLE IF EXISTS posts")
        .execute(&pool)
        .await?;
    sqlx::query(
        "
        CREATE TABLE posts (
            id UUID PRIMARY KEY,
            title TEXT NOT NULL,
            body TEXT NOT NULL,
            embedding vector(256)
        )
    ",
    )
    .execute(&pool)
    .await?;

    // 2. Load Embedder
    let model_path = std::env::var("MODEL_PATH").expect("MODEL_PATH env var is not set");
    let calc = UserPreferenceCalculator::new(&model_path)?;

    // 3. Fetch Open Source Dataset (AG News - Test Set)
    println!("Downloading dataset...");
    let dataset_url = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv";
    let response = reqwest::get(dataset_url).await?.text().await?;

    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(response.as_bytes());

    let mut posts = Vec::new();
    println!("Prefilling 1000 items into database...");

    // AG News CSV format: "Class Index","Title","Description"
    for result in reader.records().take(1000) {
        let record = result?;
        if record.len() < 3 {
            continue;
        }

        let title = record[1].trim().to_string();
        let body = record[2].trim().to_string();
        let post = Post {
            id: Uuid::new_v4().to_string(),
            title,
            body,
        };

        // Calculate embedding for pre-fill
        let interactions = vec![(post.clone(), Interaction::Liked)];
        let embedding = calc
            .calculate_preference_vector(&interactions)
            .await?
            .unwrap();

        sqlx::query(
            "INSERT INTO posts (id, title, body, embedding) VALUES ($1, $2, $3, $4::vector)",
        )
        .bind(Uuid::parse_str(&post.id)?)
        .bind(&post.title)
        .bind(&post.body)
        .bind(embedding)
        .execute(&pool)
        .await?;

        posts.push(post);
    }

    println!("Database pre-filled with {} items.", posts.len());

    // 4. Simulate User Action: Liked "World" or "Business" related topics
    // AG News tags: 1-World, 2-Sports, 3-Business, 4-Sci/Tech
    // Let's create a preference for "Sci/Tech"
    let user_history = posts
        .iter()
        .filter(|p| {
            p.title.to_lowercase().contains("tech") || p.body.to_lowercase().contains("internet")
        })
        .take(3)
        .map(|p| (p.clone(), Interaction::Liked))
        .collect::<Vec<_>>();

    if user_history.is_empty() {
        println!("No tech-related posts found in the sample, using fallback.");
        return Ok(());
    }

    let user_pref_vector = calc
        .calculate_preference_vector(&user_history)
        .await?
        .unwrap();

    // 5. Query Database for Similarity
    #[derive(sqlx::FromRow)]
    struct SearchResult {
        title: String,
        distance: f64,
    }

    let results = sqlx::query_as::<_, SearchResult>(
        "SELECT title, (embedding <=> $1::vector)::float8 as distance FROM posts ORDER BY distance ASC LIMIT 5"
    )
    .bind(user_pref_vector)
    .fetch_all(&pool)
    .await?;

    println!("\nTop 5 Recommendations for User (Preference: Tech/Internet):");
    for (i, res) in results.iter().enumerate() {
        println!("{}. {} (distance: {:.4})", i + 1, res.title, res.distance);
    }

    // 6. Assertions
    assert!(results.len() >= 1);

    Ok(())
}
