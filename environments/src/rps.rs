use std::io::{self, Write};
use rand::Rng;
use crate::Environment;

#[derive(Clone)]
pub struct RPS {
    pub current_round: usize,
    pub max_rounds: usize,
    pub player_score: f32,
    pub agent_last_move: Option<usize>,
    pub human_mode: bool,  // New field to track mode
}

impl RPS {
    // New constructor with mode
    pub fn new_with_mode(human_mode: bool) -> Self {
        RPS {
            current_round: 0,
            max_rounds: 2,
            player_score: 0.0,
            agent_last_move: None,
            human_mode,
        }
    }

    fn get_opponent_move(&self) -> usize {
        if self.human_mode {
            Self::get_human_move()
        } else {
            match self.current_round {
                0 => rand::thread_rng().gen_range(0..3),  // First round: random
                1 => self.agent_last_move.expect("Agent's last move should be recorded"), // Second round: copy agent's move
                _ => unreachable!("Game should only have 2 rounds")
            }
        }
    }

    fn calculate_round_outcome(&self, agent_move: usize, opponent_move: usize) -> f32 {
        match (agent_move, opponent_move) {
            (p, o) if p == o => 0.0,  // Draw
            (0, 2) | (1, 0) | (2, 1) => 1.0,  // Agent Wins
            _ => -1.0,  // Agent Loses
        }
    }

    fn get_move_name(move_num: usize) -> &'static str {
        match move_num {
            0 => "Rock",
            1 => "Paper",
            2 => "Scissors",
            _ => "Unknown",
        }
    }

    fn get_human_move() -> usize {
        loop {
            print!("Enter your move (0: Rock, 1: Paper, 2: Scissors): ");
            io::stdout().flush().unwrap();

            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();

            match input.trim().parse() {
                Ok(num) if num < 3 => return num,
                _ => println!("Invalid input! Please enter 0, 1, or 2."),
            }
        }
    }
}

impl Environment for RPS {
    fn new() -> Self {
        Self::new_with_mode(false)  // Default to training mode
    }

    fn num_states(&self) -> usize {
        4  // 3 states for first round + 1 initial state
    }

    fn num_actions(&self) -> usize {
        3  // Rock (0), Paper (1), Scissors (2)
    }

    fn state_id(&self) -> usize {
        match self.agent_last_move {
            None => 3,  // Initial state
            Some(last_move) => last_move,
        }
    }

    fn reset(&mut self) {
        self.current_round = 0;
        self.player_score = 0.0;
        self.agent_last_move = None;
    }

    fn is_game_over(&self) -> bool {
        self.current_round >= self.max_rounds
    }

    fn available_actions(&self) -> Vec<usize> {
        if self.is_game_over() {
            vec![]
        } else {
            vec![0, 1, 2]
        }
    }

    fn score(&self) -> f32 {
        self.player_score
    }

    fn step(&mut self, agent_action: usize) {
        if self.is_game_over() {
            panic!("We are trying to play but game is over!");
        }
        if !self.available_actions().contains(&agent_action) {
            panic!("Unauthorized action!");
        }

        let opponent_move = self.get_opponent_move();
        let round_score = self.calculate_round_outcome(agent_action, opponent_move);

        if self.human_mode {
            println!("\nRound {}:", self.current_round + 1);
            println!("AI played: {}", Self::get_move_name(agent_action));
            println!("You played: {}", Self::get_move_name(opponent_move));
            match round_score {
                1.0 => println!("AI won this round!"),
                0.0 => println!("It's a draw!"),
                -1.0 => println!("You won this round!"),
                _ => unreachable!(),
            }
        }

        if self.current_round == 0 {
            self.agent_last_move = Some(agent_action);
        }

        self.player_score += round_score;
        self.current_round += 1;
    }

    fn display(&self) {
        if self.human_mode {
            println!("Round: {}/{}", self.current_round + 1, self.max_rounds);
            println!("AI Score: {}", self.player_score);
            if let Some(last_move) = self.agent_last_move {
                println!("AI's last move: {}", Self::get_move_name(last_move));
            }
        }
    }
}