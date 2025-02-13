use algorithms::{
    RLAlgorithm,
    q_learning::QLearning,
    dyna_q::DynaQ,
    policy_iteration::PolicyIteration,
    value_iteration::ValueIteration,
    on_montecarlo_control::MonteCarloControl,
    off_montecarlo_control::OffPolicyMonteCarloControl,
    sarsa::Sarsa,
    reinforce::Reinforce,
    semi_gradient_sarsa::SemiGradientSarsa,
    dqn::DQN
};

use environments::{
    Environment,
    line_world::LineWorld,
    grid_world::GridWorld,
    rps::RPS,
    secret_env::SecretEnv
};

use std::io::{self, Write};
use serde::{Serialize, Deserialize};
use std::fs;
use std::path::Path;

#[derive(Serialize, Deserialize)]
pub enum TrainedAI {
    QLearning(QLearning),
    DynaQ(DynaQ),
    PolicyIteration(PolicyIteration),
    ValueIteration(ValueIteration),
    MonteCarloControl(MonteCarloControl),
    OffPolicyMonteCarloControl(OffPolicyMonteCarloControl),
    Sarsa(Sarsa),
    Reinforce(Reinforce),
    SemiGradientSarsa(SemiGradientSarsa),
    DQN(DQN),
}

const ALPHA: f32 = 0.01;
const EPSILON: f32 = 0.01;
const GAMMA: f32 = 0.99;
const THETA: f32 = 1e-1;
const PLANNING_STEPS: usize = 5;
const EPSILON_MC: f32 = 0.1;
const GAMMA_MC: f32 = 0.99;
const EPSILON_OFF_MC: f32 = 0.1;
const GAMMA_OFF_MC: f32 = 0.99;
const ALPHA_SARSA: f32 = 0.1;
const EPSILON_SARSA: f32 = 0.1;
const GAMMA_SARSA: f32 = 0.99;
const ALPHA_REINFORCE: f32 = 0.1;
const GAMMA_REINFORCE: f32 = 0.99;
const ALPHA_SEMI_GRADIENT_SARSA: f32 = 0.1;
const EPSILON_SEMI_GRADIENT_SARSA: f32 = 0.1;
const GAMMA_SEMI_GRADIENT_SARSA: f32 = 0.99;

const ALPHA_DQN: f32 = 0.1;
const EPSILON_DQN: f32 = 0.1;
const GAMMA_DQN: f32 = 0.99;
const MEMORY_CAPACITY_DQN: usize = 1000;
const BATCH_SIZE_DQN: usize = 32;

impl TrainedAI {
    pub fn save(&self, env_name: &str, algorithm_name: &str) -> std::io::Result<()> {
        // Create models directory if it doesn't exist
        fs::create_dir_all("models")?;

        let filename = format!("models/{}_{}.json", env_name, algorithm_name);
        let json = serde_json::to_string_pretty(self)?;
        fs::write(filename.clone(), json)?;
        println!("Model saved to {}", filename);
        Ok(())
    }

    pub fn load(env_name: &str, algorithm_name: &str) -> std::io::Result<Option<Self>> {
        let filename = format!("models/{}_{}.json", env_name, algorithm_name);

        if Path::new(&filename).exists() {
            let json = fs::read_to_string(filename.clone())?;
            let model = serde_json::from_str(&json)?;
            println!("Model loaded from {}", filename);
            Ok(Some(model))
        } else {
            println!("No saved model found at {}", filename);
            Ok(None)
        }
    }
    fn get_best_action(&self, state: usize, available_actions: &[usize]) -> usize {
        match self {
            TrainedAI::QLearning(ai) => ai.get_best_action(state, available_actions),
            TrainedAI::DynaQ(ai) => ai.get_best_action(state, available_actions),
            TrainedAI::PolicyIteration(ai) => ai.get_best_action(state, available_actions),
            TrainedAI::ValueIteration(ai) => ai.get_best_action(state, available_actions),
            TrainedAI::MonteCarloControl(ai) => ai.get_best_action(state, available_actions),
            TrainedAI::OffPolicyMonteCarloControl(ai) => ai.get_best_action(state, available_actions),
            TrainedAI::Sarsa(ai) => ai.get_best_action(state, available_actions),
            TrainedAI::Reinforce(ai) => ai.get_best_action(state, available_actions),
            TrainedAI::SemiGradientSarsa(ai) => ai.get_best_action(state, available_actions),
            TrainedAI::DQN(ai) => ai.get_best_action(state, available_actions),
        }
    }

}

fn train_ai(algorithm: &str) -> TrainedAI {
    let env = RPS::new_with_mode(false);
    println!("\nTraining AI...");

    match algorithm {
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
        "PolicyIteration" => {
            let mut ai = PolicyIteration::new(
                env.num_states(),
                env.num_actions(),
                0.99,  // gamma
                1e-6,  // theta
            );

            let num_episodes = 10000;
            let log_interval = 1000;

            println!("Training PolicyIteration for {} episodes...", num_episodes);
            let rewards = ai.train(&mut env.clone(), num_episodes);
            display_training_stats(&rewards, num_episodes, log_interval);

            TrainedAI::PolicyIteration(ai)
        },
        "ValueIteration" => {
            let mut ai = ValueIteration::new(
                env.num_states(),
                env.num_actions(),
                0.99,  // gamma
                1e-6,  // theta
            );
            let num_episodes = 10000;
            let log_interval = 1000;

            println!("Training ValueIteration for {} episodes...", num_episodes);
            let rewards = ai.train(&mut env.clone(), num_episodes);
            display_training_stats(&rewards, num_episodes, log_interval);

            TrainedAI::ValueIteration(ai)
        },
        "MonteCarloControl" => {
            let mut ai = MonteCarloControl::new(
                env.num_states(),
                env.num_actions(),
                EPSILON_MC,
                GAMMA_MC,
            );
            let num_episodes = 10000;
            let log_interval = 1000;

            println!("Training on policy MonteCarloControl for {} episodes...", num_episodes);
            let rewards = ai.train(&mut env.clone(), num_episodes);
            display_training_stats(&rewards, num_episodes, log_interval);

            TrainedAI::MonteCarloControl(ai)
        },
        "OffPolicyMonteCarloControl" => {
            let mut ai = OffPolicyMonteCarloControl::new(
                env.num_states(),
                env.num_actions(),
                EPSILON_OFF_MC,
                GAMMA_OFF_MC,
            );
            let num_episodes = 10000;
            let log_interval = 1000;

            println!("Training OffPolicyMonteCarloControl for {} episodes...", num_episodes);
            let rewards = ai.train(&mut env.clone(), num_episodes);
            display_training_stats(&rewards, num_episodes, log_interval);

            TrainedAI::OffPolicyMonteCarloControl(ai)
        },
        "Sarsa" => {
            let mut ai = Sarsa::new(
                env.num_states(),
                env.num_actions(),
                ALPHA_SARSA,
                EPSILON_SARSA,
                GAMMA_SARSA,
            );
            let num_episodes = 10000;
            let log_interval = 1000;

            println!("Training Sarsa for {} episodes...", num_episodes);
            let rewards = ai.train(&mut env.clone(), num_episodes);
            display_training_stats(&rewards, num_episodes, log_interval);

            TrainedAI::Sarsa(ai)
        },"reinforce" => {
            let mut ai = Reinforce::new(
                env.num_states(),
                env.num_actions(),
                ALPHA_REINFORCE,
                GAMMA_REINFORCE,
            );
            let num_episodes = 10000;
            let log_interval = 1000;

            println!("Training REINFORCE for {} episodes...", num_episodes);
            let rewards = ai.train(&mut env.clone(), num_episodes);
            display_training_stats(&rewards, num_episodes, log_interval);

            TrainedAI::Reinforce(ai)
        },"SemiGradientSarsa" => {
            let mut ai = SemiGradientSarsa::new(
                env.num_states() * env.num_actions(),
                ALPHA_SEMI_GRADIENT_SARSA,
                EPSILON_SEMI_GRADIENT_SARSA,
                GAMMA_SEMI_GRADIENT_SARSA,
            );
            let num_episodes = 10000;
            let log_interval = 1000;

            println!("Training SemiGradientSarsa for {} episodes...", num_episodes);
            let rewards = ai.train(&mut env.clone(), num_episodes);
            display_training_stats(&rewards, num_episodes, log_interval);

            TrainedAI::SemiGradientSarsa(ai)
        },"DQN" => {
            let mut ai = DQN::new(
                env.num_states(),
                env.num_actions(),
                ALPHA_DQN,
                EPSILON_DQN,
                GAMMA_DQN,
                MEMORY_CAPACITY_DQN,
                BATCH_SIZE_DQN
            );
            let num_episodes = 10000;
            let log_interval = 1000;

            println!("Training DQN for {} episodes...", num_episodes);
            let rewards = ai.train(&mut env.clone(), num_episodes);
            display_training_stats(&rewards, num_episodes, log_interval);

            TrainedAI::DQN(ai)
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
fn play_against_ai(algorithm: &str, env_name: &str) {

    // Try to load existing model first
    let ai = if let Ok(Some(loaded_ai)) = TrainedAI::load(env_name, algorithm) {
        println!("Using saved model...");
        loaded_ai
    } else {
        println!("No saved model found, training new model...");
        let ai = train_ai(algorithm);
        // Save the newly trained model
        if let Err(e) = ai.save(env_name, algorithm) {
            println!("Warning: Failed to save model: {}", e);
        }
        ai
    };

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

fn get_user_choice(prompt: &str, options: &[&str]) -> usize {
    loop {
        println!("\n{}", prompt);
        for (i, option) in options.iter().enumerate() {
            println!("{}. {}", i + 1, option);
        }

        print!("\nEnter your choice (1-{}): ", options.len());
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();

        if let Ok(choice) = input.trim().parse::<usize>() {
            if choice >= 1 && choice <= options.len() {
                return choice - 1;
            }
        }
        println!("Invalid choice! Please try again.");
    }
}

fn run_demonstration<T: Environment + Clone>(env_name: &str, mut env: T, algorithm: &str) {
    println!("\nDemonstrating {} with {}:", env_name, algorithm);

    let mut ai = if let Ok(Some(loaded_ai)) = TrainedAI::load(env_name, algorithm) {
        println!("Using saved model...");
        loaded_ai
    } else {
        println!("No saved model found, training new model...");
        let mut ai = match algorithm {
            "Q-Learning" => TrainedAI::QLearning(QLearning::new(
                env.num_states(),
                env.num_actions(),
                ALPHA,
                EPSILON,
                GAMMA,
            )),
            "Dyna-Q" => TrainedAI::DynaQ(DynaQ::new(
                env.num_states(),
                env.num_actions(),
                ALPHA,
                EPSILON,
                GAMMA,
                PLANNING_STEPS,
            )),
            "PolicyIteration" => TrainedAI::PolicyIteration(PolicyIteration::new(
                env.num_states(),
                env.num_actions(),
                GAMMA,
                THETA,
            )),
            "ValueIteration" => TrainedAI::ValueIteration(ValueIteration::new(
                env.num_states(),
                env.num_actions(),
                GAMMA,
                THETA,
            )),
            "MonteCarloControl" => TrainedAI::MonteCarloControl(MonteCarloControl::new(
                env.num_states(),
                env.num_actions(),
                EPSILON_MC,
                GAMMA_MC,
            )),
            "OffPolicyMonteCarloControl" => TrainedAI::OffPolicyMonteCarloControl(OffPolicyMonteCarloControl::new(
                env.num_states(),
                env.num_actions(),
                EPSILON_OFF_MC,
                GAMMA_OFF_MC,
            )),
            "Sarsa" => TrainedAI::Sarsa(Sarsa::new(
                env.num_states(),
                env.num_actions(),
                ALPHA_SARSA,
                EPSILON_SARSA,
                GAMMA_SARSA,
            )),
            "Reinforce" => TrainedAI::Reinforce(Reinforce::new(
                env.num_states(),
                env.num_actions(),
                ALPHA_REINFORCE,
                GAMMA_REINFORCE,
            )),
            "SemiGradientSarsa" => TrainedAI::SemiGradientSarsa(SemiGradientSarsa::new(
                env.num_states() * env.num_actions(),
                ALPHA_SEMI_GRADIENT_SARSA,
                EPSILON_SEMI_GRADIENT_SARSA,
                GAMMA_SEMI_GRADIENT_SARSA,
            )),
            "DQN" => TrainedAI::DQN(DQN::new(
                env.num_states(),
                env.num_actions(),
                ALPHA_DQN,
                EPSILON_DQN,
                GAMMA_DQN,
                MEMORY_CAPACITY_DQN,
                BATCH_SIZE_DQN
            )),
            _ => panic!("Unknown algorithm"),
        };

        // Train the AI
        println!("\nTraining AI...");
        let rewards = match &mut ai {
            TrainedAI::QLearning(q) => q.train(&mut env.clone(), 10000),
            TrainedAI::DynaQ(d) => d.train(&mut env.clone(), 10000),
            TrainedAI::PolicyIteration(p) => p.train(&mut env.clone(), 10000),
            TrainedAI::ValueIteration(v) => v.train(&mut env.clone(), 10000),
            TrainedAI::MonteCarloControl(c) => c.train(&mut env.clone(), 10000),
            TrainedAI::OffPolicyMonteCarloControl(c) => c.train(&mut env.clone(), 10000),
            TrainedAI::Sarsa(s) => s.train(&mut env.clone(), 10000),
            TrainedAI::Reinforce(r) => r.train(&mut env.clone(), 10000),
            TrainedAI::SemiGradientSarsa(s) => s.train(&mut env.clone(), 10000),
            TrainedAI::DQN(d) => d.train(&mut env.clone(), 10000),
        };
        display_training_stats(&rewards, 10000, 1000);

        // Save the trained model
        if let Err(e) = ai.save(env_name, algorithm) {
            println!("Warning: Failed to save model: {}", e);
        }

        ai
    };

    // Demonstrate trained behavior
    println!("\nDemonstrating trained behavior:");
    env.reset();
    println!("Initial state:");
    env.display();

    println!("\nPress Enter to see each step...");
    let mut input = String::new();

    while !env.is_game_over() {
        io::stdin().read_line(&mut input).unwrap();
        let state = env.state_id();
        let action = ai.get_best_action(state, &env.available_actions());
        env.step(action);
        env.display();
    }

    println!("Final score: {}", env.score());
}

fn main() {
    // Choose algorithm
    let algorithms = ["Q-Learning",
        "Dyna-Q",
        "PolicyIteration",
        "ValueIteration",
        "MonteCarloControl",
        "OffPolicyMonteCarloControl",
        "Sarsa",
        "Reinforce",
        "SemiGradientSarsa",
        "DQN"];
    let algorithm = algorithms[get_user_choice(
        "Choose an algorithm:",
        &algorithms
    )];

    // Choose environment
    let environments = [
        "Line World",
        "Grid World",
        "Rock Paper Scissors (Play against AI)",
        "Secret Environment 0",
        "Secret Environment 1",
        "Secret Environment 2",
        "Secret Environment 3",
    ];
    let env_choice = get_user_choice("Choose an environment:", &environments);

    match env_choice {
        0 => run_demonstration("Line World", LineWorld::new(), algorithm),
        1 => run_demonstration("Grid World", GridWorld::new(), algorithm),
        2 => {
            println!("\nStarting Rock Paper Scissors against trained {}...", algorithm);
            play_against_ai(algorithm, "RockPaperScissors");
        },
        3..=6 => {
            let env_id = env_choice - 3;
            run_demonstration(
                &format!("Secret Environment {}", env_id),
                SecretEnv::new(env_id),
                algorithm
            );
        },
        _ => unreachable!(),
    }
}