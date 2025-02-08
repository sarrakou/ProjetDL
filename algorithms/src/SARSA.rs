use environments::Environment;
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use crate::RLAlgorithm;

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

                self.q_table[state][action] += self.alpha * (
                    reward +
                        if env.is_game_over() { 0.0 }
                        else { self.gamma * self.q_table[next_state][next_action] } -
                        self.q_table[state][action]
                );

                state = next_state;
                action = next_action;
            }

            episode_rewards.push(total_reward);
        }

        episode_rewards
    }

    fn get_best_action(&self, state: usize, available_actions: &[usize]) -> usize {
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