use algorithms::{
    RLAlgorithm,
    policy_iteration::PolicyIteration,
    value_iteration::ValueIteration, // Import de ValueIteration
    on_montecarlo_control::MonteCarloControl,
    off_montecarlo_control::OffPolicyMonteCarloControl,


};

use environments::{
    Environment,
    line_world::LineWorld,
    grid_world::GridWorld,
    rps::RPS,
};

fn test_algorithm<T: Environment + Clone, A: RLAlgorithm>(algo_name: &str, mut algorithm: A, env_name: &str, mut env: T) {
    println!("\nTesting {} on {} environment:", algo_name, env_name);
    println!("Number of states: {}", env.num_states());
    println!("Number of actions: {}", env.num_actions());

    let rewards_history = algorithm.train(&mut env, 100);
    let avg_reward: f32 = rewards_history.iter().sum::<f32>() / rewards_history.len() as f32;
    println!("Average reward over training: {:.2}", avg_reward);

    env.reset();
    println!("\nTesting learned policy:");
    println!("Initial state:");
    env.display();

    let max_steps = 1000;
    let mut steps = 0;
    while !env.is_game_over() && steps < max_steps {
        let state = env.state_id();
        let available_actions = env.available_actions();

        println!("Step {}: Current state: {}, Available actions: {:?}", steps, state, available_actions);

        if available_actions.is_empty() {
            println!("No available actions! Exiting loop.");
            break;
        }

        let action = algorithm.get_best_action(state, &available_actions);
        println!("Taking action: {:?}", action);

        env.step(action);
        env.display();

        steps += 1;
    }

    if steps == max_steps {
        println!("Max steps reached, possible infinite loop.");
    }
    println!("Final score: {}", env.score());
}

fn main() {
    test_algorithm(
        "Off-policy Monte Carlo",
        OffPolicyMonteCarloControl::new(
            LineWorld::new().num_states(),
            LineWorld::new().num_actions(),
            0.1,  // epsilon
            0.99  // gamma
        ),
        "LineWorld",
        LineWorld::new()
    );

    // Test for GridWorld with Off-policy Monte Carlo
    test_algorithm(
        "Off-policy Monte Carlo",
        OffPolicyMonteCarloControl::new(
            GridWorld::new().num_states(),
            GridWorld::new().num_actions(),
            0.1,  // epsilon
            0.99  // gamma
        ),
        "GridWorld",
        GridWorld::new()
    );

    // Test for RPS with Off-policy Monte Carlo
    test_algorithm(
        "Off-policy Monte Carlo",
        OffPolicyMonteCarloControl::new(
            RPS::new().num_states(),
            RPS::new().num_actions(),
            0.05,  // epsilon réduit
            0.99   // gamma
        ),
        "RPS",
        RPS::new()
    );
}
