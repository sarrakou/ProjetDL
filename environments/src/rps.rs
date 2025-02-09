use crate::Environment;
use rand::Rng;

#[derive(Clone)]
pub struct RPS {
    current_round: usize,
    max_rounds: usize,
    player_score: f32,
    opponent_last_move: Option<usize>,
}

impl RPS {

    pub fn get_current_round(&self) -> usize {
        self.current_round
    }

    fn get_opponent_move(&self) -> usize {
        // For now, opponent plays randomly with equal probability
        let mut rng = rand::thread_rng();
        rng.random_range(0..3)
    }

    fn calculate_round_outcome(&self, player_move: usize, opponent_move: usize) -> f32 {
        match (player_move, opponent_move) {
            (p, o) if p == o => 0.0,  // Draw
            (0, 2) | (1, 0) | (2, 1) => 1.0,  // Win
            _ => -1.0,  // Loss
        }
    }
}

impl Environment for RPS {
    fn new() -> Self {
        RPS {
            current_round: 0,
            max_rounds: 2,
            player_score: 0.0,
            opponent_last_move: None,
        }
    }

    fn num_states(&self) -> usize {
        4  // 3 states for opponent's last move (R,P,S) + 1 for initial state
    }

    fn num_actions(&self) -> usize {
        3  // Rock (0), Paper (1), Scissors (2)
    }

    fn state_id(&self) -> usize {
        self.opponent_last_move.unwrap_or_else(|| 3)
    }

    fn reset(&mut self) {
        self.current_round = 0;
        self.player_score = 0.0;
        self.opponent_last_move = None;
    }

    fn is_game_over(&self) -> bool {
        self.current_round >= self.max_rounds
    }

    fn available_actions(&self) -> Vec<usize> {
        if self.is_game_over() {
            vec![]
        } else {
            vec![0, 1, 2]  // Rock, Paper, Scissors always available
        }
    }

    fn score(&self) -> f32 {
        self.player_score
    }

    fn step(&mut self, action: usize) {
        if self.is_game_over() {
            panic!("We are trying to play but game is over!");
        }
        if !self.available_actions().contains(&action) {
            panic!("Unauthorized action!");
        }

        let opponent_move = self.get_opponent_move();
        let round_score = self.calculate_round_outcome(action, opponent_move);

        self.player_score += round_score;
        self.opponent_last_move = Some(opponent_move);
        self.current_round += 1;
    }

    fn display(&self) {
        println!("Round: {}/{}", self.current_round, self.max_rounds);
        println!("Current Score: {}", self.player_score);
        if let Some(last_move) = self.opponent_last_move {
            println!("Opponent's last move: {}", match last_move {
                0 => "Rock",
                1 => "Paper",
                2 => "Scissors",
                _ => unreachable!(),
            });
        }
    }
}