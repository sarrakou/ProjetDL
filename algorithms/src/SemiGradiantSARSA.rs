use environments::Environment;
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use crate::RLAlgorithm;

pub struct SemiGradientSarsa {
    weights: Vec<f32>,
    num_features: usize,
    alpha: f32,
    epsilon: f32,
    gamma: f32,
}

impl SemiGradientSarsa {
    pub fn new(
        num_features: usize,
        alpha: f32,
        epsilon: f32,
        gamma: f32,
    ) -> Self {
        SemiGradientSarsa {
            weights: vec![0.0; num_features],
            num_features,
            alpha,
            epsilon,
            gamma,
        }
    }

    fn compute_features(&self, state: usize, action: usize) -> Vec<f32> {
        let mut features = vec![0.0; self.num_features];
        features[state * action % self.num_features] = 1.0;
        features
    }

    fn approximate_q_value(&self, state: usize, action: usize) -> f32 {
        let features = self.compute_features(state, action);
        features.iter()
            .zip(self.weights.iter())
            .map(|(f, w)| f * w)
            .sum()
    }

    fn compute_gradient(&self, state: usize, action: usize) -> Vec<f32> {
        self.compute_features(state, action)
    }
}

impl RLAlgorithm for SemiGradientSarsa {
    fn train<T: Environment>(&mut self, env: &mut T, max_episodes: usize) -> Vec<f32> {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let mut episode_rewards = Vec::new();

        for _ in 0..max_episodes {
            env.reset();
            let mut total_reward = 0.0;
            let mut state = env.state_id();

            let available_actions = env.available_actions();
            let mut action = if rand::random::<f32>() <= self.epsilon {
                *available_actions.choose(&mut rng).unwrap()
            } else {
                self.get_best_action(state, &available_actions)
            };

            while !env.is_game_over() {
                let prev_score = env.score();
                env.step(action);
                let reward = env.score() - prev_score;
                total_reward += reward;
                let next_state = env.state_id();
                let next_available_actions = env.available_actions();

                let next_action = if rand::random::<f32>() <= self.epsilon {
                    *next_available_actions.choose(&mut rng).unwrap()
                } else {
                    self.get_best_action(next_state, &next_available_actions)
                };

                let current_q = self.approximate_q_value(state, action);
                let next_q = if env.is_game_over() {
                    0.0
                } else {
                    self.approximate_q_value(next_state, next_action)
                };
                let td_error = reward + self.gamma * next_q - current_q;

                let gradient = self.compute_gradient(state, action);
                for i in 0..self.weights.len() {
                    self.weights[i] += self.alpha * td_error * gradient[i];
                }

                state = next_state;
                action = next_action;
            }

            episode_rewards.push(total_reward);
        }

        episode_rewards
    }

    fn get_best_action(&self, state: usize, available_actions: &[usize]) -> usize {
        let mut best_action = available_actions[0];
        let mut best_value = self.approximate_q_value(state, available_actions[0]);

        for &action in available_actions.iter().skip(1) {
            let value = self.approximate_q_value(state, action);
            if value > best_value {
                best_action = action;
                best_value = value;
            }
        }

        best_action
    }
}