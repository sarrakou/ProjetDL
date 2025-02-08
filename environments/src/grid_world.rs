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
        println!("Available actions for position ({}, {}):", self.pos_x, self.pos_y);

        // Up: action 0 is available if pos_y > 0
        if self.pos_y > 0 {
            actions.push(0);
        }
        // Right: action 1 is available if pos_x < size - 1
        if self.pos_x < self.size - 1 {
            actions.push(1);
        }
        // Down: action 2 is available if pos_y < size - 1
        if self.pos_y < self.size - 1 {
            actions.push(2);
        }
        // Left: action 3 is available if pos_x > 0
        if self.pos_x > 0 {
            actions.push(3);
        }

        println!("Available actions: {:?}", actions);
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
            println!("Game over! No action can be taken.");
            return;  // Simply return if the game is over.
        }

        // Check if the action is available
        if !self.available_actions().contains(&action) {
            println!("Unauthorized action! Skipping action {}.", action);
            return;  // Skip the action if it's not valid.
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
        let mut transitions = vec![vec![vec![0.0; self.num_states()]; self.num_actions()]; self.num_states()];

        // Helper function to convert state ID to grid position
        let state_to_pos = |state: usize| -> (usize, usize) {
            let y = state / self.size;
            let x = state % self.size;
            (x, y)
        };

        // Helper function to convert grid position to state ID
        let pos_to_state = |x: usize, y: usize| -> usize {
            y * self.size + x
        };

        for state in 0..self.num_states() {
            let (x, y) = state_to_pos(state);

            // Skip if current state is a terminal state
            if (x == 0 && y == 0) || (x == self.size - 1 && y == self.size - 1) {
                continue;
            }

            // Up
            if y > 0 {
                let next_state = pos_to_state(x, y - 1);
                transitions[state][0][next_state] = 1.0;
            } else {
                transitions[state][0][state] = 1.0;  // Stay in place if can't move
            }

            // Right
            if x < self.size - 1 {
                let next_state = pos_to_state(x + 1, y);
                transitions[state][1][next_state] = 1.0;
            } else {
                transitions[state][1][state] = 1.0;
            }

            // Down
            if y < self.size - 1 {
                let next_state = pos_to_state(x, y + 1);
                transitions[state][2][next_state] = 1.0;
            } else {
                transitions[state][2][state] = 1.0;
            }

            // Left
            if x > 0 {
                let next_state = pos_to_state(x - 1, y);
                transitions[state][3][next_state] = 1.0;
            } else {
                transitions[state][3][state] = 1.0;
            }
        }

        transitions
    }

    fn reward_function(&self) -> Vec<Vec<f32>> {
    let mut rewards = vec![vec![0.0; self.num_actions()]; self.num_states()];

    // Helper function to convert state ID to grid position
    let state_to_pos = |state: usize| -> (usize, usize) {
    let y = state / self.size;
    let x = state % self.size;
    (x, y)
    };

    for state in 0..self.num_states() {
    let (x, y) = state_to_pos(state);

    // Add small negative reward for each step to encourage shorter paths
    for action in 0..self.num_actions() {
    rewards[state][action] = -0.01;
    }

    // Add rewards for reaching goals
    if x == 0 && y == 0 {
    // Goal state - positive reward for all actions
    for action in 0..self.num_actions() {
    rewards[state][action] = 1.0;
    }
    } else if x == self.size - 1 && y == self.size - 1 {
    // Negative goal state - negative reward for all actions
    for action in 0..self.num_actions() {
    rewards[state][action] = -1.0;
    }
    }
    }

    rewards
    }
    fn run_policy(&mut self, policy: &[usize]) -> f32 {
        let mut total_reward = 0.0;
        let mut steps = 0;

        // Validate policy size
        if policy.len() != self.num_states() {
            println!("Error: Policy length {} doesn't match number of states {}",
                     policy.len(), self.num_states());
            return total_reward;
        }

        while !self.is_game_over() {
            let state = self.state_id();
            let available_actions = self.available_actions();

            if available_actions.is_empty() {
                println!("No available actions at state {}. Ending episode.", state);
                break;
            }

            // Get the action from the policy
            let policy_action = policy[state];

            // Choose action and apply it
            if available_actions.contains(&policy_action) {
                println!("Executing policy action {} at state {}", policy_action, state);
                self.step(policy_action);
            } else {
                let chosen_action = available_actions[0];
                println!("Policy action {} invalid at state {}. Using action {} instead.",
                         policy_action, state, chosen_action);
                self.step(chosen_action);
            }

            // Add the reward for the current step
            let step_reward = self.score();
            total_reward += step_reward;

            // Print status
            println!("Step {}: State {}, Reward {}", steps, state, step_reward);
            self.display();

            steps += 1;
            if steps > 100 {
                println!("Max steps ({}) reached, terminating episode.", steps);
                break;
            }
        }

        println!("Episode completed in {} steps with total reward: {}", steps, total_reward);
        total_reward
    }
    }


