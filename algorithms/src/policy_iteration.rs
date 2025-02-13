
use crate::RLAlgorithm;
use std::f32::EPSILON;

pub struct PolicyIteration {
    num_states: usize,
    num_actions: usize,
    gamma: f32,
    theta: f32,
    policy: Vec<usize>,  // The current policy
    value: Vec<f32>,     // The value function for each state
}

impl PolicyIteration {
    // Constructor for PolicyIteration
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

    // Function to improve the policy
    fn improve_policy(&mut self, transition_probabilities: &Vec<Vec<Vec<f32>>>, reward_function: &Vec<Vec<f32>>) {
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

        if is_policy_stable {
            return;  // The policy is stable, so we exit the function
        }
    }

    // Function to evaluate the policy
    fn evaluate_policy(&mut self, transition_probabilities: &Vec<Vec<Vec<f32>>>, reward_function: &Vec<Vec<f32>>) {
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

            // Check if the value function has converged
            if delta < self.theta {
                break;
            }
        }
    }

    // Main method for performing policy iteration
    pub fn policy_iteration(&mut self, transition_probabilities: &Vec<Vec<Vec<f32>>>, reward_function: &Vec<Vec<f32>>) {
        loop {
            // Step 1: Evaluate the policy
            self.evaluate_policy(transition_probabilities, reward_function);

            // Step 2: Improve the policy
            self.improve_policy(transition_probabilities, reward_function);

            if self.policy.iter().enumerate().all(|(state, &action)| {
                let mut max_value = f32::MIN;
                let mut best_action = 0;
                for action_candidate in 0..self.num_actions {
                    let mut value = 0.0;
                    for next_state in 0..self.num_states {
                        value += transition_probabilities[state][action_candidate][next_state]
                            * (reward_function[state][action_candidate] + self.gamma * self.value[next_state]);
                    }

                    if value > max_value {
                        max_value = value;
                        best_action = action_candidate;
                    }
                }
                action == best_action
            }) {
                break;  // The policy has converged, so we exit the loop
            }
        }
    }

    // Accessor for the learned policy
    pub fn get_policy(&self) -> &Vec<usize> {
        &self.policy
    }
}

impl Default for PolicyIteration {
    fn default() -> Self {
        Self::new(1, 1, 0.9, 1e-6)
    }
}

// Implementation of RLAlgorithm trait
impl RLAlgorithm for PolicyIteration {

    fn train<T: environments::Environment>(&mut self, env: &mut T, max_episodes: usize) -> Vec<f32> {
        let mut returns = Vec::new();


        if self.num_states != env.num_states() || self.num_actions != env.num_actions() {
            self.num_states = env.num_states();
            self.num_actions = env.num_actions();
            self.policy = vec![0; self.num_states];
            self.value = vec![0.0; self.num_states];
        }

        for _ in 0..max_episodes {

            let transition_probabilities = env.transition_probabilities();
            let reward_function = env.reward_function();


            self.policy_iteration(&transition_probabilities, &reward_function);


            let total_reward = env.run_policy(&self.policy);
            returns.push(total_reward);
        }

        returns
    }

    fn get_best_action(&self, state: usize, available_actions: &[usize]) -> usize {
        if available_actions.is_empty() {
            return 0;
        }

        let mut best_action = available_actions[0];
        let mut max_value = f32::MIN;

        for &action in available_actions {
            if self.policy[state] == action {
                return action;  // Return the action from our policy if it's available
            }

            // Fallback to using value function if policy action is not available
            if self.value[state] > max_value {
                max_value = self.value[state];
                best_action = action;
            }
        }

        best_action
    }
}