use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use environments::Environment;
use crate::RLAlgorithm;
use serde::{Serialize, Deserialize};

#[derive(Clone, Serialize, Deserialize)]
struct Transition {
    state: usize,
    action: usize,
    reward: f32,
    next_state: usize,
    done: bool,
}

#[derive(Clone, Serialize, Deserialize)]
struct ReplayMemory {
    transitions: Vec<Transition>,
    capacity: usize,
}

impl ReplayMemory {
    pub fn new(capacity: usize) -> Self {
        Self {
            transitions: Vec::with_capacity(capacity),
            capacity,
        }
    }

    /// Agrega una transición. Si se excede la capacidad, elimina la transición más antigua (FIFO).
    pub fn push(&mut self, transition: Transition) {
        if self.transitions.len() >= self.capacity {
            self.transitions.remove(0);
        }
        self.transitions.push(transition);
    }

    /// Retorna un minibatch (vector de referencias a transiciones) muestreado aleatoriamente.
    pub fn sample(&self, batch_size: usize, rng: &mut impl Rng) -> Vec<&Transition> {
        let mut indices: Vec<usize> = (0..self.transitions.len()).collect();
        indices.shuffle(rng);
        indices.into_iter().take(batch_size)
            .map(|i| &self.transitions[i])
            .collect()
    }

    pub fn len(&self) -> usize {
        self.transitions.len()
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct DQN {
    weights: Vec<Vec<f32>>,
    epsilon: f32,
    alpha: f32,
    gamma: f32,
    memory: ReplayMemory,
    batch_size: usize,
}

impl DQN {
    pub fn new(
        num_states: usize,
        num_actions: usize,
        alpha: f32,
        epsilon: f32,
        gamma: f32,
        memory_capacity: usize,
        batch_size: usize,
    ) -> Self {
        let mut rng = rand::thread_rng();
        // Inicializa la "red" con pesos aleatorios pequeños.
        let weights = (0..num_states)
            .map(|_| {
                (0..num_actions)
                    .map(|_| rng.gen_range(-0.1..0.1))
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<Vec<f32>>>();

        Self {
            weights,
            alpha,
            epsilon,
            gamma,
            memory: ReplayMemory::new(memory_capacity),
            batch_size,
        }
    }
}

impl RLAlgorithm for DQN {
    fn train<T: Environment>(&mut self, env: &mut T, max_episodes: usize) -> Vec<f32> {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let mut rewards_per_episode = Vec::with_capacity(max_episodes);

        for _ in 0..max_episodes {
            env.reset();
            let mut total_reward = 0.0;
            let mut s = env.state_id();

            while !env.is_game_over() {
                let available = env.available_actions();
                // Selección de acción: ε‑greedy.
                let a = if rng.gen::<f32>() < self.epsilon {
                    *available.choose(&mut rng).unwrap()
                } else {
                    self.get_best_action(s, &available)
                };

                // Se utiliza el mismo metodo que en Q‑Learning para calcular la recompensa.
                let prev_score = env.score();
                env.step(a);
                let r = env.score() - prev_score;
                total_reward += r;
                let s_next = env.state_id();
                let done = env.is_game_over();

                // Almacena la transición en el replay memory.
                self.memory.push(Transition {
                    state: s,
                    action: a,
                    reward: r,
                    next_state: s_next,
                    done,
                });

                s = s_next;

                // Si hay suficientes transiciones, se muestrea un minibatch y se actualiza la red.
                if self.memory.len() >= self.batch_size {
                    let minibatch = self.memory.sample(self.batch_size, &mut rng);
                    for transition in minibatch {
                        // Predicción actual
                        let q_current = self.weights[transition.state][transition.action];
                        // Calcula el valor máximo del siguiente estado (0 si es terminal).
                        let max_q_next = if transition.done {
                            0.0
                        } else {
                            *self.weights[transition.next_state]
                                .iter()
                                .max_by(|a, b| a.partial_cmp(b).unwrap())
                                .unwrap()
                        };
                        let target = transition.reward + self.gamma * max_q_next;
                        let error = target - q_current;
                        // Actualización del peso para la acción tomada.
                        self.weights[transition.state][transition.action] += self.alpha * error;
                    }
                }
            }
            rewards_per_episode.push(total_reward);
        }
        rewards_per_episode
    }

    /// Durante la evaluación, devuelve la acción con mayor Q-valor para el estado dado,
    /// restringido a las acciones disponibles.
    fn get_best_action(&self, state: usize, available_actions: &[usize]) -> usize {
        let mut best_action = available_actions[0];
        let mut best_value = self.weights[state][best_action];
        for &a in available_actions.iter().skip(1) {
            let value = self.weights[state][a];
            if value > best_value {
                best_value = value;
                best_action = a;
            }
        }
        best_action
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use environments::line_world::LineWorld;
    use crate::RLAlgorithm;

    #[test]
    fn test_dqn_initialization() {
        let env = LineWorld::new();
        // Por ejemplo: memoria con capacidad 1000 y batch_size de 32.
        let dqn = DQN::new(env.num_states(), env.num_actions(), 0.1, 0.1, 0.99, 1000, 32);
        assert_eq!(dqn.weights.len(), env.num_states());
        for row in dqn.weights.iter() {
            assert_eq!(row.len(), env.num_actions());
        }
        // El replay memory se inicializa vacío.
        assert_eq!(dqn.memory.len(), 0);
    }

    #[test]
    fn test_dqn_training() {
        let mut env = LineWorld::new();
        let mut dqn = DQN::new(env.num_states(), env.num_actions(), 0.1, 1.0, 0.99, 1000, 32);
        let rewards = dqn.train(&mut env, 100);
        // Se deben generar 100 episodios.
        assert_eq!(rewards.len(), 100);
    }
}
