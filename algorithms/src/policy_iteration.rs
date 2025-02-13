use crate::RLAlgorithm;
use environments::Environment;

use serde::{Serialize, Deserialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct PolicyIteration {
    num_states: usize,
    num_actions: usize,
    gamma: f32,
    theta: f32,
    policy: Vec<usize>,
    value: Vec<f32>,
}

impl PolicyIteration {
    pub fn new(num_states: usize, num_actions: usize, gamma: f32, theta: f32) -> Self {
        PolicyIteration {
            num_states,
            num_actions,
            gamma,
            theta,
            policy: vec![0; num_states],
            value: vec![0.0; num_states],
        }
    }

    fn improve_policy(&mut self, transition_probabilities: &[Vec<Vec<f32>>], reward_function: &[Vec<f32>]) -> bool {
        let mut is_policy_stable = true;

        for state in 0..self.num_states {
            let old_action = self.policy[state];
            let mut max_value = f32::MIN;
            let mut best_action = 0;

            for action in 0..self.num_actions {
                let mut value = 0.0;
                for next_state in 0..self.num_states {
                    value += transition_probabilities[state][action][next_state]
                        * (reward_function[state][action] + self.gamma * self.value[next_state]);
                }

                if value > max_value {
                    max_value = value;
                    best_action = action;
                }
            }

            self.policy[state] = best_action;

            if old_action != best_action {
                is_policy_stable = false;
            }
        }

        is_policy_stable
    }

    fn evaluate_policy(&mut self, transition_probabilities: &[Vec<Vec<f32>>], reward_function: &[Vec<f32>]) {
        loop {
            let mut delta: f32 = 0.0;
            for state in 0..self.num_states {
                let old_value = self.value[state];
                let action = self.policy[state];
                let mut new_value = 0.0;
                for next_state in 0..self.num_states {
                    new_value += transition_probabilities[state][action][next_state]
                        * (reward_function[state][action] + self.gamma * self.value[next_state]);
                }
                self.value[state] = new_value;
                delta = delta.max((old_value - new_value).abs());
            }

            if delta < self.theta {
                break;
            }
        }
    }

    pub fn policy_iteration(&mut self, transition_probabilities: &[Vec<Vec<f32>>], reward_function: &[Vec<f32>]) {
        loop {
            self.evaluate_policy(transition_probabilities, reward_function);
            if self.improve_policy(transition_probabilities, reward_function) {
                break;
            }
        }
    }

    pub fn get_policy(&self) -> &[usize] {
        &self.policy
    }
}

impl RLAlgorithm for PolicyIteration {
    fn train<T: Environment>(&mut self, env: &mut T, max_episodes: usize) -> Vec<f32> {
        let mut returns = Vec::new();

        if self.num_states != env.num_states() || self.num_actions != env.num_actions() {
            self.num_states = env.num_states();
            self.num_actions = env.num_actions();
            self.policy = vec![0; self.num_states];
            self.value = vec![0.0; self.num_states];
        }

        // For each episode, we'll run through the environment to collect actual experience
        for _ in 0..max_episodes {
            env.reset();
            let mut episode_reward = 0.0;

            while !env.is_game_over() {
                let state = env.state_id();
                let available_actions = env.available_actions();

                // Only choose from available actions
                if !available_actions.is_empty() {
                    let action = self.get_best_action(state, &available_actions);
                    env.step(action);
                    episode_reward += env.score();
                } else {
                    break;
                }
            }

            returns.push(episode_reward);
        }

        returns
    }

    fn get_best_action(&self, state: usize, available_actions: &[usize]) -> usize {
        if available_actions.is_empty() {
            panic!("No available actions!");
        }

        // Only consider available actions when choosing the best one
        let mut best_action = available_actions[0];
        let mut best_value = f32::MIN;

        for &action in available_actions {
            let value = if state < self.policy.len() {
                self.value[state]
            } else {
                f32::MIN
            };

            if value > best_value {
                best_value = value;
                best_action = action;
            }
        }

        best_action
    }
}