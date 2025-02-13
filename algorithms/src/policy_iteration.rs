use crate::RLAlgorithm;
use serde::{Serialize, Deserialize};
use environments::Environment;
use rand::Rng;

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
        let mut rng = rand::thread_rng();
        let policy = (0..num_states)
            .map(|_| rng.gen_range(0..num_actions))
            .collect();

        PolicyIteration {
            num_states,
            num_actions,
            gamma,
            theta,
            policy,
            value: vec![0.0; num_states],
        }
    }

    fn improve_policy<T: Environment>(&mut self, env: &mut T) -> bool {
        let mut is_policy_stable = true;
        let transitions = env.transition_probabilities();
        let rewards = env.reward_function();

        for state in 0..self.num_states {
            let old_action = self.policy[state];
            let available_actions = env.available_actions();

            let (best_action, _) = available_actions.iter()
                .map(|&action| {
                    let immediate_reward = rewards[state][action];
                    let future_value: f32 = (0..self.num_states)
                        .map(|next_state| {
                            transitions[state][action][next_state] * self.gamma * self.value[next_state]
                        })
                        .sum();
                    (action, immediate_reward + future_value)
                })
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap_or((0, f32::MIN));

            self.policy[state] = best_action;

            if old_action != best_action {
                is_policy_stable = false;
            }
        }

        is_policy_stable
    }

    fn evaluate_policy<T: Environment>(&mut self, env: &mut T) {
        let transitions = env.transition_probabilities();
        let rewards = env.reward_function();

        loop {
            let mut delta: f32 = 0.0;

            for state in 0..self.num_states {
                let old_value = self.value[state];
                let action = self.policy[state];

                let immediate_reward = rewards[state][action];
                let future_value: f32 = (0..self.num_states)
                    .map(|next_state| {
                        transitions[state][action][next_state] * self.gamma * self.value[next_state]
                    })
                    .sum();

                let new_value = immediate_reward + future_value;
                self.value[state] = new_value;

                delta = delta.max((old_value - new_value).abs());
            }

            if delta < self.theta {
                break;
            }
        }
    }

    pub fn policy_iteration<T: Environment>(&mut self, env: &mut T) {
        let mut iterations = 0;
        loop {
            self.evaluate_policy(env);
            if self.improve_policy(env) || iterations >= 100 {
                break;
            }
            iterations += 1;
        }
    }

    pub fn get_policy(&self) -> &[usize] {
        &self.policy
    }
}

impl RLAlgorithm for PolicyIteration {
    fn train<T: Environment>(&mut self, env: &mut T, max_episodes: usize) -> Vec<f32> {
        let mut returns = Vec::new();

        self.num_states = env.num_states();
        self.num_actions = env.num_actions();

        // Initialize with random policy
        let mut rng = rand::thread_rng();
        self.policy = (0..self.num_states)
            .map(|_| rng.gen_range(0..self.num_actions))
            .collect();
        self.value = vec![0.0; self.num_states];

        // Run policy iteration
        self.policy_iteration(env);

        // Collect results
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

        if state >= self.policy.len() {
            return available_actions[0];
        }

        let action = self.policy[state];
        if available_actions.contains(&action) {
            action
        } else {
            available_actions[0]
        }
    }
}