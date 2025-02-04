use crate::Environment;

pub struct LineWorld {
    pos: usize,
}

impl Environment for LineWorld {
    fn new() -> Self {
        LineWorld { pos: 2 }
    }

    fn num_states(&self) -> usize {
        5
    }

    fn num_actions(&self) -> usize {
        2
    }

    fn state_id(&self) -> usize {
        self.pos
    }

    fn reset(&mut self) {
        self.pos = 2;
    }

    fn is_game_over(&self) -> bool {
        self.pos == 0 || self.pos == 4
    }

    fn available_actions(&self) -> Vec<usize> {
        if self.is_game_over() {
            vec![]
        } else {
            vec![0, 1]
        }
    }

    fn score(&self) -> f32 {
        match self.pos {
            0 => -1.0,
            4 => 1.0,
            _ => 0.0,
        }
    }

    fn step(&mut self, action: usize) {
        if self.is_game_over() {
            panic!("We are trying to play but game is over!");
        }
        if !self.available_actions().contains(&action) {
            panic!("Unauthorized action!");
        }
        match action {
            0 => {
                self.pos -= 1;
            }
            1 => {
                self.pos += 1;
            }
            _ => unreachable!(),
        }
    }

    fn display(&self) {
        for s in 0..5 {
            if self.pos == s {
                print!("X");
            } else {
                print!("_");
            }
        }
        println!();
    }
}