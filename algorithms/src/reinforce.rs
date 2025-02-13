
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use environments::Environment;
use crate::RLAlgorithm;

pub struct Reinforce {
    policy: Vec<Vec<f32>>,
    alpha: f32,
    gamma: f32,
}

impl Reinforce {
    pub fn new(num_states: usize, num_actions: usize, alpha: f32, gamma: f32) -> Self {
        Self {
            policy: vec![vec![0.0; num_actions]; num_states],
            alpha,
            gamma,
        }
    }

    fn softmax(&self, state: usize) -> Vec<f32> {
        let logits = &self.policy[state];
        // Para mayor estabilidad se resta el máximo
        let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f32 = exp_vals.iter().sum();
        exp_vals.iter().map(|&x| x / sum).collect()
    }

    fn sample_action(&self, state: usize, rng: &mut impl Rng) -> usize {
        let probs = self.softmax(state);
        let r: f32 = rng.gen(); // Número aleatorio en [0,1)
        let mut cumulative = 0.0;
        for (i, p) in probs.iter().enumerate() {
            cumulative += *p;
            if r < cumulative {
                return i;
            }
        }
        // Por seguridad, retorna la última acción.
        self.policy[state].len() - 1
    }
}

impl RLAlgorithm for Reinforce {
    fn train<T: Environment>(&mut self, env: &mut T, max_episodes: usize) -> Vec<f32> {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let mut rewards_per_episode = Vec::with_capacity(max_episodes);

        for _ in 0..max_episodes {
            env.reset();
            let mut episode: Vec<(usize, usize, f32)> = Vec::new(); // (estado, acción, recompensa)
            let mut total_reward = 0.0;

            // Generación del episodio.
            while !env.is_game_over() {
                let state = env.state_id();
                let action = self.sample_action(state, &mut rng);
                let prev_score = env.score();
                env.step(action);
                let reward = env.score() - prev_score;  // La recompensa se calcula como la diferencia en score.
                total_reward += reward;
                episode.push((state, action, reward));
            }
            rewards_per_episode.push(total_reward);

            // Actualización de la política para cada paso del episodio.
            let n = episode.len();
            for t in 0..n {
                let mut G = 0.0;
                let mut discount = 1.0;
                for k in (t + 1)..n {
                    G += discount * episode[k].2;
                    discount *= self.gamma;
                }
                // Factor de actualización: α · (γ^t) · G.
                let update_factor = self.alpha * self.gamma.powi(t as i32) * G;
                let (state, action, _) = episode[t];
                let probs = self.softmax(state);
                // Actualiza para cada acción: suma el término de actualización.
                for a in 0..self.policy[state].len() {
                    let grad = if a == action { 1.0 } else { 0.0 } - probs[a];
                    self.policy[state][a] += update_factor * grad;
                }
            }
        }
        rewards_per_episode
    }

    /// Durante la evaluación, devuelve la acción con mayor probabilidad para el estado dado.
    fn get_best_action(&self, state: usize, available_actions: &[usize]) -> usize {
        let probs = self.softmax(state);
        let mut best_action = available_actions[0];
        let mut best_prob = probs[best_action];
        for &a in available_actions.iter() {
            if probs[a] > best_prob {
                best_prob = probs[a];
                best_action = a;
            }
        }
        best_action
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use environments::line_world::LineWorld;
    use crate::RLAlgorithm;

    #[test]
    fn test_reinforce_initialization() {
        let env = LineWorld::new();
        let reinforce = Reinforce::new(env.num_states(), env.num_actions(), 0.1, 0.99);
        assert_eq!(reinforce.policy.len(), env.num_states());
        for row in reinforce.policy.iter() {
            assert_eq!(row.len(), env.num_actions());
        }
    }

    #[test]
    fn test_reinforce_training() {
        let mut env = LineWorld::new();
        let mut reinforce = Reinforce::new(env.num_states(), env.num_actions(), 0.1, 0.99);
        let rewards = reinforce.train(&mut env, 100);
        // Se deben generar 100 episodios.
        assert_eq!(rewards.len(), 100);
    }
}
