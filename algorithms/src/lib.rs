pub mod q_learning;
//pub mod dyna_q;

pub mod policy_iteration;

pub trait RLAlgorithm {
    fn train<T: environments::Environment>(&mut self, env: &mut T, max_episodes: usize) -> Vec<f32>;
    fn get_best_action(&self, state: usize, available_actions: &[usize]) -> usize;
}

