use environments::Environment;
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use crate::RLAlgorithm;

#[derive(Clone)]
pub struct QLearning {
    q_table: Vec<Vec<f32>>,
    alpha: f32,
    epsilon: f32,
    gamma: f32,
}

impl QLearning {
    pub fn new(num_states: usize, num_actions: usize, alpha: f32, epsilon: f32, gamma: f32) -> Self {
        let mut q_table = Vec::new();
        for _ in 0..num_states {
            q_table.push(vec![0.0; num_actions]);
        }

        QLearning {
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

impl RLAlgorithm for QLearning {
    fn train<T: Environment>(&mut self, env: &mut T, max_episodes: usize) -> Vec<f32> {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let mut episode_rewards = Vec::new();

        for _ in 0..max_episodes {
            env.reset();
            let mut total_reward = 0.0;
            let mut s = env.state_id();

            while !env.is_game_over() {
                // Get available actions and choose one using epsilon-greedy policy
                let aa = env.available_actions();
                let a = if rand::random::<f32>() <= self.epsilon {
                    *aa.choose(&mut rng).unwrap()
                } else {
                    self.get_best_action(s, &aa)
                };

                // Take action and observe reward and next state
                let prev_score = env.score();
                env.step(a);
                let r = env.score() - prev_score;
                total_reward += r;

                // Get next state and its available actions
                let s_next = env.state_id();

                // Calculate target Q-value
                let max_q_next = if env.is_game_over() {
                    0.0
                } else {
                    let aa_next = env.available_actions();
                    aa_next.iter()
                        .map(|&a| self.q_table[s_next][a])
                        .fold(f32::MIN, f32::max)
                };

                // Update Q-value
                self.q_table[s][a] += self.alpha * (r + self.gamma * max_q_next - self.q_table[s][a]);

                s = s_next;
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

    #[test]
    fn test_qlearning_initialization() {
        let env = LineWorld::new();
        let ql = QLearning::new(env.num_states(), env.num_actions(), 0.1, 0.1, 0.99);

        // Check Q-table dimensions
        assert_eq!(ql.q_table.len(), env.num_states());
        for row in ql.q_table.iter() {
            assert_eq!(row.len(), env.num_actions());
        }
    }

    #[test]
    fn test_qlearning_training() {
        let mut env = LineWorld::new();
        let mut ql = QLearning::new(env.num_states(), env.num_actions(), 0.1, 1.0, 0.99);

        // Train for a few episodes
        let rewards = ql.train(&mut env, 100);

        // Check that we got the expected number of rewards
        assert_eq!(rewards.len(), 100);
    }
}