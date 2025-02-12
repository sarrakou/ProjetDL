use algorithms::{
    RLAlgorithm,
    q_learning::QLearning,
    dyna_q::DynaQ ,
};

use environments::{
    Environment,
    line_world::LineWorld,
    grid_world::GridWorld,
    rps::RPS,
};

use std::io::{self, Write};

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
pub enum TrainedAI {
    QLearning(QLearning),
    DynaQ(DynaQ),
}

impl TrainedAI {
    fn get_best_action(&self, state: usize, available_actions: &[usize]) -> usize {
        match self {
            TrainedAI::QLearning(ai) => ai.get_best_action(state, available_actions),
            TrainedAI::DynaQ(ai) => ai.get_best_action(state, available_actions),
        }
    }
}

fn train_ai(algorithm: &str) -> TrainedAI {
    let env = RPS::new_with_mode(false);
    println!("\nTraining AI...");

    match algorithm.to_lowercase().as_str() {
        "q-learning" => {
            let mut ai = QLearning::new(
                env.num_states(),
                env.num_actions(),
                0.001,    // alpha
                0.001,    // epsilon
                0.99,   // gamma
            );

            let num_episodes = 10000;
            let log_interval = 1000;  // Log every 1000 episodes

            println!("Training Q-Learning for {} episodes...", num_episodes);
            let rewards = ai.train(&mut env.clone(), num_episodes);

            // Calculate and display statistics for each interval
            for i in (0..num_episodes).step_by(log_interval) {
                let end = (i + log_interval).min(num_episodes);
                let interval_rewards = &rewards[i..end];

                // Calculate statistics
                let avg_reward: f32 = interval_rewards.iter().sum::<f32>() / interval_rewards.len() as f32;
                let wins = interval_rewards.iter().filter(|&&r| r > 0.0).count();
                let draws = interval_rewards.iter().filter(|&&r| r == 0.0).count();
                let losses = interval_rewards.iter().filter(|&&r| r < 0.0).count();
                let winrate = (wins as f32 * 100.0) / interval_rewards.len() as f32;

                println!(
                    "Episodes {}-{}: Avg Reward: {:.2}, Winrate: {:.1}% (W: {}, D: {}, L: {})",
                    i, end-1, avg_reward, winrate, wins, draws, losses
                );
            }

            TrainedAI::QLearning(ai)
        },
        "dyna-q" => {
            let mut ai = DynaQ::new(
                env.num_states(),
                env.num_actions(),
                0.001,    // alpha
                0.001,    // epsilon
                0.99,   // gamma
                5,      // planning steps
            );

            let num_episodes = 10000;
            let log_interval = 1000;

            println!("Training Dyna-Q for {} episodes...", num_episodes);
            let rewards = ai.train(&mut env.clone(), num_episodes);

            // Calculate and display statistics for each interval
            for i in (0..num_episodes).step_by(log_interval) {
                let end = (i + log_interval).min(num_episodes);
                let interval_rewards = &rewards[i..end];

                // Calculate statistics
                let avg_reward: f32 = interval_rewards.iter().sum::<f32>() / interval_rewards.len() as f32;
                let wins = interval_rewards.iter().filter(|&&r| r > 0.0).count();
                let draws = interval_rewards.iter().filter(|&&r| r == 0.0).count();
                let losses = interval_rewards.iter().filter(|&&r| r < 0.0).count();
                let winrate = (wins as f32 * 100.0) / interval_rewards.len() as f32;

                println!(
                    "Episodes {}-{}: Avg Reward: {:.2}, Winrate: {:.1}% (W: {}, D: {}, L: {})",
                    i, end-1, avg_reward, winrate, wins, draws, losses
                );
            }

            TrainedAI::DynaQ(ai)
        },
        _ => panic!("Unknown algorithm: {}", algorithm),
    }
}

fn play_against_ai(algorithm: &str) {
    let ai = train_ai(algorithm);
    // Human mode: human plays as opponent
    let mut game = RPS::new_with_mode(true);

    println!("\nWelcome to Rock Paper Scissors vs AI ({})!", algorithm);
    println!("You'll play {} rounds.", game.max_rounds);
    println!("The AI has been trained on 10000 games against an opponent that:");
    println!("- Plays randomly in the first round");
    println!("- Copies the AI's first move in the second round");
    println!("\nNow you'll play against the trained AI!");

    while !game.is_game_over() {
        game.display();
        let state = game.state_id();
        let ai_action = ai.get_best_action(state, &game.available_actions());
        game.step(ai_action);
    }

    println!("\nGame Over!");
    println!("Final AI score: {}", game.score());
    match game.score() {
        s if s > 0.0 => println!("AI won the game!"),
        s if s < 0.0 => println!("You won the game!"),
        _ => println!("It's a tie!"),
    }
}

fn main() {
    println!("Choose your opponent:");
    println!("1. Q-Learning AI");
    println!("2. Dyna-Q AI");

    let choice = loop {
        print!("Enter your choice (1 or 2): ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();

        match input.trim() {
            "1" => break "q-learning",
            "2" => break "dyna-q",
            _ => println!("Invalid choice! Please enter 1 or 2."),
        }
    };

    play_against_ai(choice);
}
/*fn main() {
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
} */