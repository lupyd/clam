pub mod persistent_calc;

use anyhow::Result;
use model2vec_rs::model::StaticModel;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Interaction {
    Liked,
    Disliked,
    TimeSpent(f32), // Time spent in seconds
    Shared,
    Commented,
}

impl Interaction {
    pub fn weight(&self) -> f32 {
        match self {
            Interaction::Liked => 1.0,
            Interaction::Disliked => -1.0,
            Interaction::TimeSpent(seconds) => (seconds / 60.0).min(5.0), // Cap weight at 5 minutes
            Interaction::Shared => 2.0,
            Interaction::Commented => 1.5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Post {
    pub id: String,
    pub title: String,
    pub body: String,
}

pub struct UserPreferenceCalculator {
    model: StaticModel,
}

impl UserPreferenceCalculator {
    pub fn new(model_path: &str) -> Result<Self> {
        let model = StaticModel::from_pretrained(model_path, None, None, None)?;

        Ok(Self { model })
    }

    /// Calculates the user preference vector by weighted average of post embeddings.
    pub async fn calculate_preference_vector(
        &self,
        interactions: &[(Post, Interaction)],
    ) -> Result<Option<Vec<f32>>> {
        if interactions.is_empty() {
            return Ok(None);
        }

        // Collect texts to embed
        let texts: Vec<String> = interactions
            .iter()
            .map(|(post, _)| format!("{}\n{}", post.title, post.body))
            .collect();

        // Generate embeddings for all posts in a batch
        let embeddings = self.model.encode(&texts);

        if embeddings.is_empty() {
            return Ok(None);
        }

        let dim = embeddings[0].len();

        if dim == 0 {
            return Ok(None);
        }

        let mut result = vec![0.0; dim];
        let mut total_absolute_weight = 0.0;

        for (embedding, (_, interaction)) in embeddings.iter().zip(interactions.iter()) {
            let weight = interaction.weight();
            for (i, &val) in embedding.iter().enumerate() {
                result[i] += val * weight;
            }
            total_absolute_weight += weight.abs();
        }

        if total_absolute_weight > 0.0 {
            for val in result.iter_mut() {
                *val /= total_absolute_weight;
            }
        }

        Ok(Some(result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculator_initialization() {
        let model_path = std::env::var("MODEL_PATH").expect("MODEL_PATH env var is not set");
        let calc = UserPreferenceCalculator::new(&model_path);
        assert!(calc.is_ok());
    }

    #[tokio::test]
    async fn test_preference_calculation() -> Result<()> {
        let model_path = std::env::var("MODEL_PATH").expect("MODEL_PATH env var is not set");
        let calc = UserPreferenceCalculator::new(&model_path)?;

        let post1 = Post {
            id: "1".into(),
            title: "Programming in Rust".into(),
            body: "Rust is a fast and safe systems programming language.".into(),
        };
        let post2 = Post {
            id: "2".into(),
            title: "Baking bread".into(),
            body: "How to make the perfect sourdough starter at home.".into(),
        };

        let interactions = vec![(post1, Interaction::Liked), (post2, Interaction::Disliked)];

        let result = calc.calculate_preference_vector(&interactions).await?;

        eprintln!("{:?}", result);
        assert!(result.is_some());
        let vector = result.unwrap();

        assert!(!vector.is_empty());
        Ok(())
    }

    #[test]
    fn test_interaction_weights() {
        assert_eq!(Interaction::Liked.weight(), 1.0);
        assert_eq!(Interaction::Disliked.weight(), -1.0);
        assert_eq!(Interaction::Shared.weight(), 2.0);
        assert_eq!(Interaction::Commented.weight(), 1.5);
        assert_eq!(Interaction::TimeSpent(30.0).weight(), 0.5);
        assert_eq!(Interaction::TimeSpent(600.0).weight(), 5.0); // Capped
    }

    #[tokio::test]
    async fn test_empty_interactions() -> Result<()> {
        let model_path = std::env::var("MODEL_PATH").expect("MODEL_PATH env var is not set");
        let calc = UserPreferenceCalculator::new(&model_path)?;
        assert!(calc.calculate_preference_vector(&[]).await?.is_none());
        Ok(())
    }
}
