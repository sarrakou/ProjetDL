use environments::Environment;
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Serialize, Deserialize};
use crate::RLAlgorithm;

#[derive(Clone, Serialize, Deserialize)]
pub struct Sarsa {
    q_table: Vec<Vec<f32>>,
    alpha: f32,
    epsilon: f32,
    gamma: f32,
}

impl Sarsa {
    pub fn new(
        num_states: usize,
        num_actions: usize,
        alpha: f32,
        epsilon: f32,
        gamma: f32,
    ) -> Self {
        let mut q_table = Vec::new();
        for _ in 0..num_states {
            q_table.push(vec![0.0; num_actions]);
        }

        Sarsa {
            q_table,
            alpha,
            epsilon,
            gamma,
        }
    }

    pub fn get_q_table(&self) -> &Vec<Vec<f32>> {
        &self.q_table
    }
}

impl RLAlgorithm for Sarsa {
    fn train<T: Environment>(&mut self, env: &mut T, max_episodes: usize) -> Vec<f32> {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let mut episode_rewards = Vec::new();

        for episode in 0..max_episodes {
            env.reset();
            let mut total_reward = 0.0;
            let mut state = env.state_id();

            // Protection contre un état sans actions disponibles
            let available_actions = env.available_actions();
            if available_actions.is_empty() {
                continue;
            }

            // Premier choix d'action
            let mut action = if rand::random::<f32>() <= self.epsilon {
                *available_actions.choose(&mut rng).unwrap()
            } else {
                self.get_best_action(state, &available_actions)
            };

            let mut steps = 0;
            let max_steps = 25; // Limite de pas par épisode

            // Boucle principale d'apprentissage
            while !env.is_game_over() && steps < max_steps {
                let prev_state = state;
                let prev_action = action;

                // Exécuter l'action
                let prev_score = env.score();
                env.step(action);
                let reward = env.score() - prev_score;
                total_reward += reward;

                state = env.state_id();
                steps += 1;

                if env.is_game_over() {
                    // Mise à jour finale de Q
                    self.q_table[prev_state][prev_action] += self.alpha * (
                        reward - self.q_table[prev_state][prev_action]
                    );
                    break;
                }

                // Obtenir les actions disponibles pour le nouvel état
                let next_available_actions = env.available_actions();
                if next_available_actions.is_empty() {
                    break;
                }

                // Sélectionner la prochaine action (ε-greedy)
                let next_action = if rand::random::<f32>() <= self.epsilon {
                    *next_available_actions.choose(&mut rng).unwrap()
                } else {
                    self.get_best_action(state, &next_available_actions)
                };

                // Mise à jour de Q avec la règle SARSA
                self.q_table[prev_state][prev_action] += self.alpha * (
                    reward +
                        self.gamma * self.q_table[state][next_action] -
                        self.q_table[prev_state][prev_action]
                );

                action = next_action;
            }

            episode_rewards.push(total_reward);
        }

        episode_rewards
    }

    fn get_best_action(&self, state: usize, available_actions: &[usize]) -> usize {
        if available_actions.is_empty() {
            panic!("Pas d'actions disponibles pour l'état {}", state);
        }

        let mut best_action = available_actions[0];
        let mut best_value = self.q_table[state][available_actions[0]];

        for &action in available_actions.iter().skip(1) {
            let value = self.q_table[state][action];
            if value > best_value {
                best_action = action;
                best_value = value;
            }
        }

        best_action
    }
}