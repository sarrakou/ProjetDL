use rand::Rng;
use std::collections::HashMap;
use environments::Environment;
use super::RLAlgorithm;


pub struct MonteCarloControl {
    num_states: usize,
    num_actions: usize,
    epsilon: f32,
    gamma: f32,
    q_values: Vec<Vec<f32>>,
    returns: Vec<Vec<Vec<f32>>>,
    policy: Vec<Vec<f32>>,
}

impl MonteCarloControl {
    pub fn new(num_states: usize, num_actions: usize, epsilon: f32, gamma: f32) -> Self {
        let q_values = vec![vec![0.0; num_actions]; num_states];
        let returns = vec![vec![Vec::new(); num_actions]; num_states];
        let policy = vec![vec![1.0 / num_actions as f32; num_actions]; num_states];

        Self {
            num_states,
            num_actions,
            epsilon,
            gamma,
            q_values,
            returns,
            policy,
        }
    }

    fn generate_episode<T: Environment>(&self, env: &mut T) -> Vec<(usize, usize, f32)> {
        let mut episode = Vec::new();
        env.reset();

        while !env.is_game_over() {
            let state = env.state_id();
            let available_actions = env.available_actions();

            // Select action according to ε-soft policy
            let action = self.select_action(state, &available_actions);

            let prev_score = env.score();
            env.step(action);
            let reward = env.score() - prev_score;

            episode.push((state, action, reward));
        }

        episode
    }

    fn select_action(&self, state: usize, available_actions: &[usize]) -> usize {
        let mut rng = rand::thread_rng();

        if rng.gen::<f32>() < self.epsilon {
            // Random action with probability ε
            available_actions[rng.gen_range(0..available_actions.len())]
        } else {
            // Greedy action with probability 1-ε
            let mut best_action = available_actions[0];
            let mut best_value = self.q_values[state][best_action];

            for &action in available_actions.iter().skip(1) {
                let value = self.q_values[state][action];
                if value > best_value {
                    best_value = value;
                    best_action = action;
                }
            }
            best_action
        }
    }

    fn update_policy(&mut self, state: usize) {
        // Find the best action
        let mut best_action = 0;
        let mut best_value = self.q_values[state][0];

        for action in 1..self.num_actions {
            let value = self.q_values[state][action];
            if value > best_value {
                best_value = value;
                best_action = action;
            }
        }

        // Update policy probabilities
        let epsilon_prob = self.epsilon / self.num_actions as f32;
        for action in 0..self.num_actions {
            self.policy[state][action] = if action == best_action {
                1.0 - self.epsilon + epsilon_prob
            } else {
                epsilon_prob
            };
        }
    }
}

impl RLAlgorithm for MonteCarloControl {
    fn train<T: Environment>(&mut self, env: &mut T, max_episodes: usize) -> Vec<f32> {
        let mut rewards_history = Vec::new();

        for _ in 0..max_episodes {
            // Generate episode using current policy
            let episode = self.generate_episode(env);
            let total_reward: f32 = episode.iter().map(|(_,_,r)| r).sum();
            rewards_history.push(total_reward);

            // Process episode
            let mut g = 0.0;
            let mut visited = HashMap::new();

            for (t, (state, action, reward)) in episode.iter().enumerate().rev() {
                g = self.gamma * g + reward;

                // First-visit check
                if !visited.contains_key(&(*state, *action)) {
                    visited.insert((*state, *action), true);

                    // Update returns and Q-value
                    self.returns[*state][*action].push(g);
                    self.q_values[*state][*action] = self.returns[*state][*action]
                        .iter()
                        .sum::<f32>() / self.returns[*state][*action].len() as f32;

                    // Update policy
                    self.update_policy(*state);
                }
            }
        }

        rewards_history
    }

    fn get_best_action(&self, state: usize, available_actions: &[usize]) -> usize {
        let mut best_action = available_actions[0];
        let mut best_value = self.q_values[state][best_action];

        for &action in available_actions.iter().skip(1) {
            let value = self.q_values[state][action];
            if value > best_value {
                best_value = value;
                best_action = action;
            }
        }

        best_action
    }

    fn get_policy(&self) -> Vec<usize> {
        todo!()
    }
}