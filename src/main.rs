use algorithms::{RLAlgorithm, q_learning::QLearning};
use environments::{
    Environment,
    line_world::LineWorld,
    grid_world::GridWorld,
    rps::RPS,
};

fn test_environment<T: Environment>(env_name: &str, mut env: T) {
    println!("\nTesting {} environment:", env_name);
    println!("Number of states: {}", env.num_states());
    println!("Number of actions: {}", env.num_actions());

    // Initialize Q-learning with parameters
    let mut q_learning = QLearning::new(
        env.num_states(),
        env.num_actions(),
        0.1,    // alpha (learning rate)
        0.1,    // epsilon (exploration rate)
        0.99,   // gamma (discount factor)
    );

    // Train for 1000 episodes
    let rewards = q_learning.train(&mut env, 1000);

    // Print training statistics
    let avg_reward: f32 = rewards.iter().sum::<f32>() / rewards.len() as f32;
    println!("Average reward over training: {:.2}", avg_reward);

    // Test learned policy
    env.reset();
    println!("\nTesting learned policy:");
    println!("Initial state:");
    env.display();

    while !env.is_game_over() {
        let state = env.state_id();
        let action = q_learning.get_best_action(state, &env.available_actions());
        env.step(action);
        env.display();
    }
    println!("Final score: {}", env.score());
}

fn main() {
    // Test LineWorld
    test_environment("LineWorld", LineWorld::new());

    // Test GridWorld
    test_environment("GridWorld", GridWorld::new());

    // Test RPS
    test_environment("Rock Paper Scissors", RPS::new());
}