pub mod q_learning;
pub mod dyna_q;
mod sarsa;
mod SemiGradiantSarsa;

pub trait RLAlgorithm: Send {
    fn train<T: environments::Environment + Clone>(&mut self, env: &mut T, max_episodes: usize) -> Vec<f32>;
    fn get_best_action(&self, state: usize, available_actions: &[usize]) -> usize;
}