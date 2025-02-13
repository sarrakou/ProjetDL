pub mod q_learning;
pub mod dyna_q;
mod sarsa;
mod SemiGradiantSarsa;
mod off_montecarlo_control;
mod on_montecarlo_control;
mod value_iteration;
mod dynamic_programming;
mod policy_iteration;

pub trait RLAlgorithm: Send {
    fn train<T: environments::Environment + Clone>(&mut self, env: &mut T, max_episodes: usize) -> Vec<f32>;
    fn get_best_action(&self, state: usize, available_actions: &[usize]) -> usize;
}