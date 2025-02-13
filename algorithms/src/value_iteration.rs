use std::f32;
use environments::Environment;
use crate::RLAlgorithm;

use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueIteration {
    num_states: usize,
    num_actions: usize,
    gamma: f32,
    theta: f32,
    values: Vec<f32>,
    policy: Vec<usize>,
    q_values: Vec<Vec<f32>>,
}

impl ValueIteration {
    pub fn new(num_states: usize, num_actions: usize, gamma: f32, theta: f32) -> Self {
        ValueIteration {
            num_states,
            num_actions,
            gamma,
            theta,
            values: vec![0.0; num_states],
            policy: vec![0; num_states],
            q_values: vec![vec![0.0; num_actions]; num_states],
        }
    }

    pub fn value_iteration(&mut self, transition_probs: &[Vec<Vec<f32>>], rewards: &[Vec<f32>]) {
        let max_iterations = 1000;
        let mut iterations = 0;

        loop {
            let mut delta: f32 = 0.0;
            iterations += 1;

            for s in 0..self.num_states {
                let v = self.values[s];

                // Calculate Q(s,a) for each action
                for a in 0..self.num_actions {
                    let mut q_value = 0.0;
                    for s_next in 0..self.num_states {
                        q_value += transition_probs[s][a][s_next] *
                            (rewards[s][a] + self.gamma * self.values[s_next]);
                    }
                    self.q_values[s][a] = q_value;
                }

                // Update V(s) with maximum Q-value
                if let Some(max_q) = self.q_values[s].iter().copied().max_by(|a, b| a.partial_cmp(b).unwrap()) {
                    self.values[s] = max_q;
                    // Update policy with action that maximizes Q
                    if let Some((best_action, _)) = self.q_values[s].iter()
                        .enumerate()
                        .max_by(|(_, q1), (_, q2)| q1.partial_cmp(q2).unwrap()) {
                        self.policy[s] = best_action;
                    }
                }

                delta = delta.max((v - self.values[s]).abs());
            }

            if delta < self.theta || iterations >= max_iterations {
                break;
            }
        }
    }

    pub fn get_values(&self) -> &[f32] {
        &self.values
    }

    pub fn get_q_values(&self) -> &[Vec<f32>] {
        &self.q_values
    }
}

impl RLAlgorithm for ValueIteration {
    fn train<T: Environment>(&mut self, env: &mut T, max_episodes: usize) -> Vec<f32> {
        let mut returns = Vec::new();

        if self.num_states != env.num_states() || self.num_actions != env.num_actions() {
            self.num_states = env.num_states();
            self.num_actions = env.num_actions();
            self.values = vec![0.0; self.num_states];
            self.policy = vec![0; self.num_states];
            self.q_values = vec![vec![0.0; self.num_actions]; self.num_states];
        }

        for _ in 0..max_episodes {
            env.reset();
            let mut episode_reward = 0.0;

            while !env.is_game_over() {
                let state = env.state_id();
                let available_actions = env.available_actions();

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

        // Select action with best Q-value among available actions only
        available_actions.iter()
            .max_by(|&&a1, &&a2| {
                let q1 = if state < self.q_values.len() { self.q_values[state][a1] } else { f32::NEG_INFINITY };
                let q2 = if state < self.q_values.len() { self.q_values[state][a2] } else { f32::NEG_INFINITY };
                q1.partial_cmp(&q2).unwrap()
            })
            .copied()
            .unwrap_or(available_actions[0])
    }
}