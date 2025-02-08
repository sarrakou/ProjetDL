use algorithms::{
    RLAlgorithm,
    policy_iteration::PolicyIteration,
};

use environments::{
    Environment,
    line_world::LineWorld,
    grid_world::GridWorld,  // Import de GridWorld
    rps::RPS,
};

fn test_algorithm<T: Environment + Clone>(algo_name: &str, mut algorithm: PolicyIteration, env_name: &str, mut env: T) {
    println!("\nTesting {} on {} environment:", algo_name, env_name);
    println!("Number of states: {}", env.num_states());
    println!("Number of actions: {}", env.num_actions());

    let p = env.transition_probabilities();  // Transition probabilities
    let r = env.reward_function();  // Reward function
    algorithm.policy_iteration(&p, &r);

    let avg_reward: f32 = env.run_policy(&algorithm.get_policy());  // Run policy
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

        // Vérification de la validité de l'indice de l'état avant d'accéder à la politique
        if state >= algorithm.get_policy().len() {
            println!("Invalid state index: {}", state);
            break;
        }

        let action = algorithm.get_policy()[state];
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
    // Test de l'algorithme sur LineWorld
    /*test_algorithm(
        "Policy Iteration",
        PolicyIteration::new(
            LineWorld::new().num_states(),
            LineWorld::new().num_actions(),  // Pass num_actions
            0.99,  // gamma (discount factor)
            1e-6   // theta (convergence threshold)
        ),
        "LineWorld",
        LineWorld::new()
    );*/

    // Test de l'algorithme sur GridWorld
    /*test_algorithm(
        "Policy Iteration",
        PolicyIteration::new(
            GridWorld::new().num_states(),
            GridWorld::new().num_actions(),  // Pass num_actions
            0.99,  // gamma (discount factor)
            1e-6   // theta (convergence threshold)
        ),
        "GridWorld",
        GridWorld::new()
    );*/
    test_algorithm(
        "Policy Iteration",
        PolicyIteration::new(
            RPS::new().num_states(),
            RPS::new().num_actions(),
            0.99,
            1e-6
        ),
        "RPS",
        RPS::new()
    );
}
