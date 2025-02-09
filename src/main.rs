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
    monty_hall_paradox1::MontyHall,
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

    // Obtain the learned policy from the algorithm
    let policy = algorithm.get_policy();

    // Run the policy
    let total_reward = env.run_policy(&policy);
    println!("Total reward from running the policy: {}", total_reward);

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
            MontyHall::new().num_states(),
            MontyHall::new().num_actions(),
            0.1,  // epsilon
            0.99  // gamma
        ),
        "Monty Hall",
        MontyHall::new()
    );
}
