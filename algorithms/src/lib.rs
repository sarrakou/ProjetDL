pub mod q_learning;
//pub mod dyna_q;

pub mod policy_iteration;
mod dynamic_programming;
pub mod value_iteration;
pub mod on_montecarlo_control;
pub mod off_montecarlo_control;

pub trait RLAlgorithm {
    fn train<T: environments::Environment>(&mut self, env: &mut T, max_episodes: usize) -> Vec<f32>;
    fn get_best_action(&self, state: usize, available_actions: &[usize]) -> usize;
    fn get_policy(&self) -> Vec<usize>;
}

