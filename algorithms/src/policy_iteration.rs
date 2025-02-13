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

            // The policy has converged if no action changes in `improve_policy`
            // Since we are already checking for policy stability inside `improve_policy`,
            // we don't need to repeat the check here unless handling edge cases
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
