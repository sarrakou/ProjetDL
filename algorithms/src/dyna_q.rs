use environments::Environment;
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use crate::RLAlgorithm;
use std::collections::HashMap;

#[derive(Clone)]
pub struct DynaQ {
    q_table: Vec<Vec<f32>>,
    model: HashMap<(usize, usize), (f32, usize)>, // (state, action) -> (reward, next_state)
    alpha: f32,
    epsilon: f32,
    gamma: f32,
    planning_steps: usize,  // number of model-based updates (n in the algorithm)
}

impl DynaQ {
    pub fn new(
        num_states: usize,
        num_actions: usize,
        alpha: f32,
        epsilon: f32,
        gamma: f32,
        planning_steps: usize,
    ) -> Self {
        let mut q_table = Vec::new();
        for _ in 0..num_states {
            q_table.push(vec![0.0; num_actions]);
        }

        DynaQ {
            q_table,
            model: HashMap::new(),
            alpha,
            epsilon,
            gamma,
            planning_steps,
        }
    }

    fn update_q_value(&mut self, state: usize, action: usize, reward: f32, next_state: usize, next_actions: &[usize]) {
        let max_q_next = if next_actions.is_empty() {
            0.0
        } else {
            next_actions.iter()
                .map(|&a| self.q_table[next_state][a])
                .fold(f32::MIN, f32::max)
        };

        self.q_table[state][action] += self.alpha * (
            reward + self.gamma * max_q_next - self.q_table[state][action]
        );
    }

    fn planning_step(&mut self, rng: &mut Xoshiro256PlusPlus) {
        if self.model.is_empty() {
            return;
        }

        // Sample a random state-action pair that we've seen before
        let &(state, action) = self.model.keys().choose(rng).unwrap();
        let &(reward, next_state) = self.model.get(&(state, action)).unwrap();

        // Get available actions for the next state
        let next_actions: Vec<usize> = (0..self.q_table[next_state].len()).collect();

        // Update Q-value using the model
        self.update_q_value(state, action, reward, next_state, &next_actions);
    }

    pub fn get_q_table(&self) -> &Vec<Vec<f32>> {
        &self.q_table
    }
}

impl RLAlgorithm for DynaQ {
    fn train<T: Environment>(&mut self, env: &mut T, max_episodes: usize) -> Vec<f32> {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let mut episode_rewards = Vec::new();

        for _ in 0..max_episodes {
            env.reset();
            let mut total_reward = 0.0;

            while !env.is_game_over() {
                let state = env.state_id();
                let available_actions = env.available_actions();

                // Choose action using epsilon-greedy policy
                let action = if rand::random::<f32>() <= self.epsilon {
                    *available_actions.choose(&mut rng).unwrap()
                } else {
                    self.get_best_action(state, &available_actions)
                };

                // Take action in environment
                let prev_score = env.score();
                env.step(action);
                let reward = env.score() - prev_score;
                total_reward += reward;
                let next_state = env.state_id();

                // Update Q-value using real experience
                self.update_q_value(
                    state,
                    action,
                    reward,
                    next_state,
                    &env.available_actions(),
                );

                // Store transition in model (assuming deterministic environment)
                self.model.insert((state, action), (reward, next_state));

                // Perform planning steps
                for _ in 0..self.planning_steps {
                    self.planning_step(&mut rng);
                }
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

#[cfg(test)]
mod tests {
    use super::*;
    use environments::line_world::LineWorld;
    use environments::grid_world::GridWorld;

    #[test]
    fn test_dynaq_initialization() {
        let env = LineWorld::new();
        let dynaq = DynaQ::new(
            env.num_states(),
            env.num_actions(),
            0.1,
            0.1,
            0.99,
            5,
        );

        // Check Q-table dimensions
        assert_eq!(dynaq.q_table.len(), env.num_states());
        for row in dynaq.q_table.iter() {
            assert_eq!(row.len(), env.num_actions());
        }

        // Check model is initially empty
        assert!(dynaq.model.is_empty());
    }

    #[test]
    fn test_model_updates() {
        let mut env = LineWorld::new();
        let mut dynaq = DynaQ::new(
            env.num_states(),
            env.num_actions(),
            0.1,
            1.0,  // Always explore for predictable testing
            0.99,
            5,
        );

        // Take one step and check model
        let state = env.state_id();
        let action = 0;  // Move left
        env.step(action);
        let next_state = env.state_id();

        // Train for one step
        dynaq.train(&mut env, 1);

        // Check that the transition was stored in the model
        assert!(dynaq.model.contains_key(&(state, action)));
        if let Some(&(_, stored_next_state)) = dynaq.model.get(&(state, action)) {
            assert_eq!(stored_next_state, next_state);
        }
    }

    #[test]
    fn test_planning_steps() {
        let mut env = GridWorld::new();
        let planning_steps = 10;
        let mut dynaq = DynaQ::new(
            env.num_states(),
            env.num_actions(),
            0.1,
            0.1,
            0.99,
            planning_steps,
        );

        // Store initial Q-values
        let mut initial_q_values = Vec::new();
        for row in dynaq.q_table.iter() {
            initial_q_values.extend(row.iter());
        }

        // Train for one episode
        dynaq.train(&mut env, 1);

        // Check that Q-values have been updated more times than just from direct experience
        let mut final_q_values = Vec::new();
        for row in dynaq.q_table.iter() {
            final_q_values.extend(row.iter());
        }

        // Count how many Q-values changed
        let changes = initial_q_values.iter()
            .zip(final_q_values.iter())
            .filter(|(&a, &b)| (a - b).abs() > 1e-5)
            .count();

        // Should have more changes than just from direct experience
        // due to planning steps
        assert!(changes > 0);
    }
}