use algorithms::{
    RLAlgorithm,
    q_learning::QLearning,
    dyna_q::DynaQ ,
    SARSA::Sarsa,
    SemiGradiantSARSA::SemiGradientSarsa,
};

use environments::{
    Environment,
    line_world::LineWorld,
    grid_world::GridWorld,
    rps::RPS,
};



fn test_algorithm<T: Environment + Clone>(
    algo_name: &str,
    mut algorithm: impl RLAlgorithm,
    env_name: &str,
    mut env: T
) {
    println!("\nTesting {} on {} environment:", algo_name, env_name);
    println!("Number of states: {}", env.num_states());
    println!("Number of actions: {}", env.num_actions());

    let rewards = algorithm.train(&mut env, 1000);

    let avg_reward: f32 = rewards.iter().sum::<f32>() / rewards.len() as f32;
    println!("Average reward over training: {:.2}", avg_reward);

    env.reset();
    println!("\nTesting learned policy:");
    println!("Initial state:");
    env.display();

    while !env.is_game_over() {
        let state = env.state_id();
        let action = algorithm.get_best_action(state, &env.available_actions());
        env.step(action);
        env.display();
    }
    println!("Final score: {}", env.score());
}

fn compare_algorithms<T: Environment + Clone>(
    algo1_name: &str,
    mut algo1: impl RLAlgorithm,
    algo2_name: &str,
    mut algo2: impl RLAlgorithm,
    env_name: &str,
    env: T,
    num_episodes: usize,
    num_runs: usize,
) {
    println!("\nComparing {} vs {} on {}:", algo1_name, algo2_name, env_name);
    println!("Running {} episodes {} times", num_episodes, num_runs);

    let mut algo1_results = Vec::new();
    let mut algo2_results = Vec::new();

    for run in 0..num_runs {
        println!("\nRun {}/{}", run + 1, num_runs);

        // Algorithm 1
        let mut env_copy = env.clone();
        let algo1_rewards = algo1.train(&mut env_copy, num_episodes);
        algo1_results.push(algo1_rewards);

        // Algorithm 2
        let mut env_copy = env.clone();
        let algo2_rewards = algo2.train(&mut env_copy, num_episodes);
        algo2_results.push(algo2_rewards);
    }

    // Calculate average rewards for each episode
    println!("\nResults summary:");
    for episode in 0..num_episodes {
        let algo1_avg: f32 = algo1_results.iter()
            .map(|run| run[episode])
            .sum::<f32>() / num_runs as f32;

        let algo2_avg: f32 = algo2_results.iter()
            .map(|run| run[episode])
            .sum::<f32>() / num_runs as f32;

        if episode % (num_episodes / 10) == 0 {
            println!(
                "Episode {}: {} avg reward: {:.2}, {} avg reward: {:.2}",
                episode, algo1_name, algo1_avg, algo2_name, algo2_avg
            );
        }
    }
}
fn main() {
    let q_learning = QLearning::new(
        LineWorld::new().num_states(),
        LineWorld::new().num_actions(),
        0.1,
        0.1,
        0.99,
    );
    test_algorithm("Q-Learning", q_learning, "LineWorld", LineWorld::new());

    let q_learning = QLearning::new(
        GridWorld::new().num_states(),
        GridWorld::new().num_actions(),
        0.1,
        0.1,
        0.99,
    );
    test_algorithm("Q-Learning", q_learning, "GridWorld", GridWorld::new());

    let q_learning = QLearning::new(
        RPS::new().num_states(),
        RPS::new().num_actions(),
        0.1,
        0.1,
        0.99,
    );
    test_algorithm("Q-Learning", q_learning, "Rock Paper Scissors", RPS::new());

    // Test Dyna-Q on all environments
    let dyna_q = DynaQ::new(
        LineWorld::new().num_states(),
        LineWorld::new().num_actions(),
        0.1,
        0.1,
        0.99,
        5,
    );
    test_algorithm("Dyna-Q", dyna_q, "LineWorld", LineWorld::new());

    let dyna_q = DynaQ::new(
        GridWorld::new().num_states(),
        GridWorld::new().num_actions(),
        0.1,
        0.1,
        0.99,
        5,
    );
    test_algorithm("Dyna-Q", dyna_q, "GridWorld", GridWorld::new());

    let dyna_q = DynaQ::new(
        RPS::new().num_states(),
        RPS::new().num_actions(),
        0.1,
        0.1,
        0.99,
        5,
    );
    test_algorithm("Dyna-Q", dyna_q, "Rock Paper Scissors", RPS::new());

    // Compare algorithms on each environment
    let q_learning = QLearning::new(
        LineWorld::new().num_states(),
        LineWorld::new().num_actions(),
        0.1, 0.1, 0.99
    );
    let dyna_q = DynaQ::new(
        LineWorld::new().num_states(),
        LineWorld::new().num_actions(),
        0.1, 0.1, 0.99, 5
    );
    compare_algorithms(
        "Q-Learning",
        q_learning,
        "Dyna-Q",
        dyna_q,
        "LineWorld",
        LineWorld::new(),
        1000,
        5
    );

    let q_learning = QLearning::new(
        GridWorld::new().num_states(),
        GridWorld::new().num_actions(),
        0.1, 0.1, 0.99
    );
    let dyna_q = DynaQ::new(
        GridWorld::new().num_states(),
        GridWorld::new().num_actions(),
        0.1, 0.1, 0.99, 5
    );
    compare_algorithms(
        "Q-Learning",
        q_learning,
        "Dyna-Q",
        dyna_q,
        "GridWorld",
        GridWorld::new(),
        1000,
        5
    );

    let q_learning = QLearning::new(
        RPS::new().num_states(),
        RPS::new().num_actions(),
        0.1, 0.1, 0.99
    );
    let dyna_q = DynaQ::new(
        RPS::new().num_states(),
        RPS::new().num_actions(),
        0.1, 0.1, 0.99, 5
    );
    compare_algorithms(
        "Q-Learning",
        q_learning,
        "Dyna-Q",
        dyna_q,
        "Rock Paper Scissors",
        RPS::new(),
        1000,
        5
    );

    println!("\n=== Tests de SARSA ===");

    let sarsa = Sarsa::new(
        LineWorld::new().num_states(),
        LineWorld::new().num_actions(),
        0.1,
        0.1,
        0.99,
    );
    test_algorithm("SARSA", sarsa, "LineWorld", LineWorld::new());

    let sarsa = Sarsa::new(
        GridWorld::new().num_states(),
        GridWorld::new().num_actions(),
        0.1,
        0.1,
        0.99,
    );
    test_algorithm("SARSA", sarsa, "GridWorld", GridWorld::new());

    let sarsa = Sarsa::new(
        RPS::new().num_states(),
        RPS::new().num_actions(),
        0.1,
        0.1,
        0.99,
    );
    test_algorithm("SARSA", sarsa, "Rock Paper Scissors", RPS::new());

    println!("\n=== Tests de Semi-Gradient SARSA ===");

    let semi_gradient = SemiGradientSarsa::new(
        100,
        0.1,
        0.1,
        0.99,
    );
    test_algorithm("Semi-Gradient SARSA", semi_gradient, "LineWorld", LineWorld::new());

    let semi_gradient = SemiGradientSarsa::new(
        200,
        0.1,
        0.1,
        0.99,
    );
    test_algorithm("Semi-Gradient SARSA", semi_gradient, "GridWorld", GridWorld::new());

    let semi_gradient = SemiGradientSarsa::new(
        50,
        0.1,
        0.1,
        0.99,
    );
    test_algorithm("Semi-Gradient SARSA", semi_gradient, "Rock Paper Scissors", RPS::new());

    println!("\n=== Comparaisons des algorithmes ===");

    let sarsa = Sarsa::new(
        LineWorld::new().num_states(),
        LineWorld::new().num_actions(),
        0.1, 0.1, 0.99
    );
    let semi_gradient = SemiGradientSarsa::new(
        100, 0.1, 0.1, 0.99
    );
    compare_algorithms(
        "SARSA",
        sarsa,
        "Semi-Gradient SARSA",
        semi_gradient,
        "LineWorld",
        LineWorld::new(),
        1000,
        5
    );

    let sarsa = Sarsa::new(
        LineWorld::new().num_states(),
        LineWorld::new().num_actions(),
        0.1, 0.1, 0.99
    );
    let q_learning = QLearning::new(
        LineWorld::new().num_states(),
        LineWorld::new().num_actions(),
        0.1, 0.1, 0.99
    );
    compare_algorithms(
        "SARSA",
        sarsa,
        "Q-Learning",
        q_learning,
        "LineWorld",
        LineWorld::new(),
        1000,
        5
    );

}