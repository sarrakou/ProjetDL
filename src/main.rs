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
            display_training_stats(&rewards, num_episodes, log_interval);

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
            display_training_stats(&rewards, num_episodes, log_interval);

            TrainedAI::DynaQ(ai)
        },
        _ => panic!("Unknown algorithm: {}", algorithm),
    }
}
fn display_training_stats(rewards: &[f32], num_episodes: usize, log_interval: usize) {
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