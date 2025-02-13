use rand::Rng;
use std::collections::{HashMap, HashSet};
use crate::RLAlgorithm;

pub struct OffPolicyMonteCarloControl {
    num_states: usize,
    num_actions: usize,
    q_values: Vec<Vec<f32>>,
    c_values: Vec<Vec<f32>>,
    policy: Vec<usize>,
    epsilon: f32,
    gamma: f32,
    min_epsilon: f32,
    epsilon_decay: f32,
    visit_counts: HashMap<(usize, usize), usize>, // (state, action) pairs
    episode_states: HashSet<usize>,               // États visités dans l'épisode actuel
}

impl OffPolicyMonteCarloControl {
    pub fn new(num_states: usize, num_actions: usize, epsilon: f32, gamma: f32) -> Self {
        Self {
            num_states,
            num_actions,
            q_values: vec![vec![0.0; num_actions]; num_states],
            c_values: vec![vec![0.0; num_actions]; num_states],
            policy: (0..num_states).map(|_| 0).collect(),
            epsilon,
            gamma,
            min_epsilon: 0.01,
            epsilon_decay: 0.995,
            visit_counts: HashMap::new(),
            episode_states: HashSet::new(),
        }
    }

    fn calculate_ucb(&self, state: usize, action: usize, total_visits: usize) -> f32 {
        let state_action_visits = *self.visit_counts.get(&(state, action)).unwrap_or(&0) as f32;
        let exploitation = self.q_values[state][action];

        if state_action_visits == 0.0 {
            return f32::INFINITY;
        }

        // UCB1 formula avec bonus pour les états non visités dans l'épisode actuel
        let exploration = (2.0 * (total_visits as f32).ln() / state_action_visits).sqrt();
        let novelty_bonus = if self.episode_states.contains(&state) { 0.0 } else { 1.0 };

        exploitation + exploration + novelty_bonus
    }

    fn behavior_policy(&mut self, state: usize, available_actions: &[usize]) -> usize {
        if available_actions.is_empty() {
            return 0;
        }

        let mut rng = rand::thread_rng();
        let total_visits: usize = self.visit_counts.values().sum();

        if rng.gen::<f32>() < self.epsilon {
            // Utiliser UCB pour l'exploration informée
            available_actions.iter()
                .max_by(|&&a1, &&a2| {
                    let ucb1 = self.calculate_ucb(state, a1, total_visits);
                    let ucb2 = self.calculate_ucb(state, a2, total_visits);
                    ucb1.partial_cmp(&ucb2).unwrap_or(std::cmp::Ordering::Equal)
                })
                .copied()
                .unwrap_or(available_actions[0])
        } else {
            self.get_best_action(state, available_actions)
        }
    }

    fn behavior_probability(&self, action: usize, state: usize, available_actions: &[usize]) -> f32 {
        if available_actions.is_empty() {
            return 1.0;
        }

        let best_action = self.get_best_action(state, available_actions);
        if action == best_action {
            1.0 - self.epsilon + (self.epsilon / available_actions.len() as f32)
        } else {
            self.epsilon / available_actions.len() as f32
        }
    }

    fn generate_episode<T: environments::Environment>(
        &mut self,
        env: &mut T,
    ) -> Vec<(usize, usize, f32)> {
        let mut episode = Vec::new();
        self.episode_states.clear();
        env.reset();

        let max_steps = 100;
        let mut steps = 0;
        let mut cumulative_reward = 0.0;

        while !env.is_game_over() && steps < max_steps {
            let state = env.state_id();
            let available_actions = env.available_actions();

            if available_actions.is_empty() {
                break;
            }

            self.episode_states.insert(state);
            let action = self.behavior_policy(state, &available_actions);

            // Mettre à jour les compteurs de visites
            *self.visit_counts.entry((state, action)).or_insert(0) += 1;

            let old_score = env.score();
            env.step(action);
            let reward = env.score() - old_score;

            cumulative_reward += reward;

            // Ajuster la récompense en fonction du progrès
            let adjusted_reward = if self.episode_states.len() > steps {
                // Récompense bonus pour explorer de nouveaux états
                reward + 0.1
            } else {
                // Pénalité pour revisiter les mêmes états
                reward - 0.1 * (steps - self.episode_states.len()) as f32
            };

            episode.push((state, action, adjusted_reward));
            steps += 1;
        }

        if steps == max_steps {
            // Pénaliser les épisodes qui atteignent la limite de pas
            episode.last_mut().map(|(_s, _a, r)| *r -= 1.0);
        }

        episode
    }
}

impl RLAlgorithm for OffPolicyMonteCarloControl {
    fn train<T: environments::Environment>(&mut self, env: &mut T, max_episodes: usize) -> Vec<f32> {
        let mut rewards_history = Vec::new();
        let mut current_epsilon = self.epsilon;

        for _ in 0..max_episodes {
            let episode = self.generate_episode(env);
            let mut g = 0.0;
            let mut w = 1.0;

            for (t, &(state, action, reward)) in episode.iter().enumerate().rev() {
                g = self.gamma * g + reward;

                self.c_values[state][action] += w;
                let c = self.c_values[state][action];

                // Mise à jour plus stable avec un learning rate adaptatif
                let alpha = 1.0 / (c + 1.0);
                self.q_values[state][action] += alpha * w * (g - self.q_values[state][action]);

                let available_actions = env.available_actions();
                self.policy[state] = self.get_best_action(state, &available_actions);

                if action != self.policy[state] {
                    break;
                }

                if !available_actions.is_empty() {
                    w /= self.behavior_probability(action, state, &available_actions);
                }
            }

            // Décroissance d'epsilon avec un plancher
            current_epsilon = (current_epsilon * self.epsilon_decay).max(self.min_epsilon);
            self.epsilon = current_epsilon;

            let total_reward: f32 = episode.iter().map(|&(_, _, r)| r).sum();
            rewards_history.push(total_reward);
        }

        rewards_history
    }

    fn get_best_action(&self, state: usize, available_actions: &[usize]) -> usize {
        if available_actions.is_empty() {
            return 0;
        }

        available_actions.iter()
            .max_by(|&&a1, &&a2| {
                let v1 = self.q_values[state][a1];
                let v2 = self.q_values[state][a2];
                v1.partial_cmp(&v2).unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
            .unwrap_or(available_actions[0])
    }
}