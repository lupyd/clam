use anyhow::Result;
use chrono::{DateTime, Utc};
use embed_anything::embed_query;
use embed_anything::embeddings::embed::Embedder;
use sqlx::{Pool, Sqlite};

use crate::{Interaction, Post};

pub struct PersistentPreferenceCalculator {
    pool: Pool<Sqlite>,
    model: Embedder,
    // decay half-life in seconds
    half_life_secs: f64,
}

impl PersistentPreferenceCalculator {
    /// Creates a new persistent user preference calculator.
    ///
    /// `pool` should be a connected SQLite pool.
    /// `half_life_secs` defines the halflife for interaction decay (e.g. 86400.0 for 1 day).
    pub async fn new(pool: Pool<Sqlite>, model: Embedder, half_life_secs: f64) -> Result<Self> {
        // Initialize table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT PRIMARY KEY,
                preference_vector TEXT NOT NULL,
                total_weight REAL NOT NULL,
                last_updated_at DATETIME NOT NULL
            )
            "#,
        )
        .execute(&pool)
        .await?;

        Ok(Self {
            pool,
            model,
            half_life_secs,
        })
    }

    /// Creates a new persistent user preference calculator by loading the model from path.
    pub async fn new_with_path(pool: Pool<Sqlite>, model_path: &str, half_life_secs: f64) -> Result<Self> {
        let model = embed_anything::embeddings::local::model2vec::Model2VecEmbedder::new(model_path, None, None)
            .map_err(|e| anyhow::anyhow!("Failed to initialize model: {}", e))?;
        let embedder = Embedder::Text(embed_anything::embeddings::embed::TextEmbedder::Model2Vec(
            model.into(),
        ));
        Self::new(pool, embedder, half_life_secs).await
    }

    /// Fetches the user's current preference vector
    pub async fn get_user_preference(&self, user_id: &str) -> Result<Option<Vec<f32>>> {
        let row: Option<(String, f64, DateTime<Utc>)> = sqlx::query_as(
            r#"
            SELECT preference_vector, total_weight, last_updated_at
            FROM user_preferences
            WHERE user_id = $1
            "#,
        )
        .bind(user_id)
        .fetch_optional(&self.pool)
        .await?;

        if let Some((vec_json, _weight, _updated_at)) = row {
            let vec: Vec<f32> = serde_json::from_str(&vec_json)?;
            Ok(Some(vec))
        } else {
            Ok(None)
        }
    }

    /// Records a new user interaction and updates their preference vector, with time-based decay.
    pub async fn add_interaction(
        &self,
        user_id: &str,
        post: &Post,
        interaction: Interaction,
    ) -> Result<()> {
        let now = Utc::now();
        let texts = vec![format!("{}\n{}", post.title, post.body)];
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let embeddings = embed_query(&text_refs, &self.model, None).await?;

        if embeddings.is_empty() {
            return Ok(());
        }

        let incoming_vec = embeddings[0].embedding.to_dense()?;
        let weight = interaction.weight();

        let row: Option<(String, f64, DateTime<Utc>)> = sqlx::query_as(
            r#"
            SELECT preference_vector, total_weight, last_updated_at
            FROM user_preferences
            WHERE user_id = $1
            "#,
        )
        .bind(user_id)
        .fetch_optional(&self.pool)
        .await?;

        let (mut new_vec, total_weight, last_updated) = match row {
            Some((vec_json, tw, updated_at)) => {
                let vec: Vec<f32> = serde_json::from_str(&vec_json)?;
                (vec, tw, updated_at)
            }
            None => (vec![0.0; incoming_vec.len()], 0.0, now),
        };

        let elapsed = now.signed_duration_since(last_updated).num_seconds() as f64;
        let elapsed = elapsed.max(0.0);

        // Exponential decay: Total weight decays by a half-life factor based on elapsed time.
        // E.g., if elapsed == half_life_secs, total_weight halves since the last update.
        let decay_factor = (0.5_f64).powf(elapsed / self.half_life_secs);

        let decaying_weight = total_weight * decay_factor;
        let new_total_weight = decaying_weight + weight.abs() as f64;

        if new_total_weight > 0.0 {
            for (i, v) in new_vec.iter_mut().enumerate() {
                // Reconstruct the unnormalized decayed sum, add new interaction, and renormalize.
                let old_sum = *v as f64 * decaying_weight;
                let new_sum = old_sum + (incoming_vec[i] * weight) as f64;
                *v = (new_sum / new_total_weight) as f32;
            }
        } else {
            new_vec.fill(0.0);
        }

        let vec_json = serde_json::to_string(&new_vec)?;

        sqlx::query(
            r#"
            INSERT INTO user_preferences (user_id, preference_vector, total_weight, last_updated_at)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT(user_id) DO UPDATE SET
                preference_vector = excluded.preference_vector,
                total_weight = excluded.total_weight,
                last_updated_at = excluded.last_updated_at
            "#,
        )
        .bind(user_id)
        .bind(vec_json)
        .bind(new_total_weight)
        .bind(now)
        .execute(&self.pool)
        .await?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use embed_anything::embeddings::local::model2vec::Model2VecEmbedder;

    #[tokio::test]
    async fn test_persistent_calculator() -> Result<()> {
        let model_path = std::env::var("MODEL_PATH").expect("MODEL_PATH env var is not set");
        let model = Model2VecEmbedder::new(&model_path, None, None)?;
        let embedder = Embedder::Text(embed_anything::embeddings::embed::TextEmbedder::Model2Vec(
            model.into(),
        ));

        // Use in-memory SQLite DB
        let pool = sqlx::sqlite::SqlitePoolOptions::new()
            .connect("sqlite::memory:")
            .await?;
        let calc = PersistentPreferenceCalculator::new(pool, embedder, 86400.0).await?;

        let post1 = Post {
            id: "1".into(),
            title: "Rust programming".into(),
            body: "Rust's async/await system is robust.".into(),
        };

        // Initially no preference
        assert!(calc.get_user_preference("user_test").await?.is_none());

        // Add interaction
        calc.add_interaction("user_test", &post1, Interaction::Liked)
            .await?;

        let pref1 = calc.get_user_preference("user_test").await?;
        assert!(pref1.is_some());

        // We could theoretically check decay by overriding time, but at least this runs without errors.
        Ok(())
    }
}
