pub mod line_world;
pub mod grid_world;
pub mod rps;
pub mod secret_env;

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use line_world::LineWorld;

    #[test]
    fn test_line_world() {
        let mut env = LineWorld::new();
        assert_eq!(env.state_id(), 2);  // Should start in middle position
        assert_eq!(env.available_actions(), vec![0, 1]);
        assert!(!env.is_game_over());
    }
}

#[test]
fn test_grid_world() {
    use grid_world::GridWorld;

    let mut env = GridWorld::new();
    assert_eq!(env.state_id(), 5);  // Should start at (1,1) in a 4x4 grid

    // Test available actions from starting position
    let actions = env.available_actions();
    assert!(actions.contains(&0));  // Can go up
    assert!(actions.contains(&1));  // Can go right
    assert!(actions.contains(&2));  // Can go down
    assert!(actions.contains(&3));  // Can go left

    // Test movement
    env.step(0);  // Go up
    assert_eq!(env.state_id(), 1);  // Now at (1,0)

    // Test goal state
    env.step(3);  // Go left to reach goal
    assert!(env.is_game_over());
    assert_eq!(env.score(), 1.0);
}

#[test]
fn test_rps() {
    use rps::RPS;

    let mut env = RPS::new();
    assert_eq!(env.state_id(), 3);  // Initial state
    assert_eq!(env.current_round, 0);
    assert_eq!(env.available_actions(), vec![0, 1, 2]);

    // Play one round
    env.step(0);  // Play Rock
    assert!(!env.is_game_over());  // Game shouldn't be over after one round
    assert_eq!(env.current_round, 1);

    // Play second round
    env.step(1);  // Play Paper
    assert!(env.is_game_over());  // Game should be over after two rounds

    // Test reset
    env.reset();
    assert_eq!(env.current_round, 0);
    assert_eq!(env.score(), 0.0);
}