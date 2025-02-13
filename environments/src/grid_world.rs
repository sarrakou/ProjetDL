use crate::Environment;

#[derive(Clone)]
pub struct GridWorld {
    pos_x: usize,
    pos_y: usize,
    size: usize,
}

impl Environment for GridWorld {
    fn new() -> Self {
        GridWorld {
            pos_x: 1,
            pos_y: 1,
            size: 4,  // 4x4 grid
        }
    }

    fn num_states(&self) -> usize {
        self.size * self.size
    }

    fn num_actions(&self) -> usize {
        4  // Up, Right, Down, Left
    }

    fn state_id(&self) -> usize {
        self.pos_y * self.size + self.pos_x
    }

    fn reset(&mut self) {
        self.pos_x = 1;
        self.pos_y = 1;
    }

    fn is_game_over(&self) -> bool {
        (self.pos_x == 0 && self.pos_y == 0) ||  // Goal state (top-left)
            (self.pos_x == self.size - 1 && self.pos_y == self.size - 1)  // Negative goal state (bottom-right)
    }

    fn available_actions(&self) -> Vec<usize> {
        if self.is_game_over() {
            return vec![];
        }

        let mut actions = Vec::new();

        // Up
        if self.pos_y > 0 {
            actions.push(0);
        }
        // Right
        if self.pos_x < self.size - 1 {
            actions.push(1);
        }
        // Down
        if self.pos_y < self.size - 1 {
            actions.push(2);
        }
        // Left
        if self.pos_x > 0 {
            actions.push(3);
        }

        actions
    }

    fn score(&self) -> f32 {
        if self.pos_x == 0 && self.pos_y == 0 {
            1.0  // Reached goal state
        } else if self.pos_x == self.size - 1 && self.pos_y == self.size - 1 {
            -1.0  // Reached negative goal state
        } else {
            0.0  // Still playing
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
            0 => self.pos_y -= 1,  // Up
            1 => self.pos_x += 1,  // Right
            2 => self.pos_y += 1,  // Down
            3 => self.pos_x -= 1,  // Left
            _ => unreachable!(),
        }
    }

    fn display(&self) {
        for y in 0..self.size {
            for x in 0..self.size {
                if x == self.pos_x && y == self.pos_y {
                    print!("X");
                } else if x == 0 && y == 0 {
                    print!("G");  // Goal
                } else if x == self.size - 1 && y == self.size - 1 {
                    print!("B");  // Bad state
                } else {
                    print!(".");
                }
                print!(" ");
            }
            println!();
        }
    }

    fn transition_probabilities(&self) -> Vec<Vec<Vec<f32>>> {
        let mut probs = vec![vec![vec![0.0; self.num_states()]; self.num_actions()]; self.num_states()];

        for y in 0..self.size {
            for x in 0..self.size {
                let state = y * self.size + x;

                // For each state, calculate transitions for each action
                for action in 0..4 {
                    // Check if action is valid in this state
                    let mut next_x = x;
                    let mut next_y = y;

                    match action {
                        0 if y > 0 => next_y -= 1,         // Up
                        1 if x < self.size - 1 => next_x += 1,  // Right
                        2 if y < self.size - 1 => next_y += 1,  // Down
                        3 if x > 0 => next_x -= 1,         // Left
                        _ => continue,  // Invalid action
                    }

                    let next_state = next_y * self.size + next_x;
                    probs[state][action][next_state] = 1.0;  // Deterministic transition
                }
            }
        }

        probs
    }

    fn reward_function(&self) -> Vec<Vec<f32>> {
        let mut rewards = vec![vec![0.0; self.num_actions()]; self.num_states()];

        for y in 0..self.size {
            for x in 0..self.size {
                let state = y * self.size + x;

                // Define rewards for actions from this state
                for action in 0..4 {
                    let mut next_x = x;
                    let mut next_y = y;

                    match action {
                        0 if y > 0 => next_y -= 1,         // Up
                        1 if x < self.size - 1 => next_x += 1,  // Right
                        2 if y < self.size - 1 => next_y += 1,  // Down
                        3 if x > 0 => next_x -= 1,         // Left
                        _ => continue,  // Invalid action
                    }

                    // Goal state (top-left corner)
                    if next_x == 0 && next_y == 0 {
                        rewards[state][action] = 1.0;
                    }
                    // Negative goal state (bottom-right corner)
                    else if next_x == self.size - 1 && next_y == self.size - 1 {
                        rewards[state][action] = -1.0;
                    }
                }
            }
        }

        rewards
    }
}