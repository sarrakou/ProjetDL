pub mod line_world;
pub mod grid_world;
pub mod rps;
pub mod secret_env;
pub mod monty_hall_paradox1;
pub mod monty_hall_paradox2;

pub trait Environment {
    fn new() -> Self;
    fn num_states(&self) -> usize;
    fn num_actions(&self) -> usize;
    fn state_id(&self) -> usize;
    fn reset(&mut self);
    fn is_game_over(&self) -> bool;
    fn available_actions(&self) -> Vec<usize>;
    fn score(&self) -> f32;
    fn step(&mut self, action: usize);
    fn display(&self);

    //méthodes pour PolicyIteration
    fn transition_probabilities(&self) -> Vec<Vec<Vec<f32>>> {
        vec![vec![vec![0.0; self.num_states()]; self.num_actions()]; self.num_states()]
    }
    fn reward_function(&self) -> Vec<Vec<f32>> {
        vec![vec![0.0; self.num_actions()]; self.num_states()]
    }
    fn run_policy(&mut self, policy: &[usize]) -> f32 {
        self.reset();
        let mut total_reward = 0.0;
        while !self.is_game_over() {
            let state = self.state_id();
            let action = policy[state];
            self.step(action);
            total_reward += self.score();
        }
        total_reward
    }
}