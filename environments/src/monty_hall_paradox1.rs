use rand::Rng;
use crate::Environment;

#[derive(Clone)]
pub struct MontyHall {
    winning_door: usize,
    chosen_door: Option<usize>,
    revealed_door: Option<usize>,
    final_choice: Option<usize>,
}

impl Environment for MontyHall {
    fn new() -> Self {
        let mut rng = rand::thread_rng();
        MontyHall {
            winning_door: rng.gen_range(0..3),
            chosen_door: None,
            revealed_door: None,
            final_choice: None,
        }
    }

    fn num_states(&self) -> usize {
        3 * 3 * 3  // 3 portes * 3 choix initiaux * 3 portes révélées
    }

    fn num_actions(&self) -> usize {
        3 // Choisir une porte A, B ou C (0, 1, 2) puis garder ou changer
    }

    fn state_id(&self) -> usize {
        let chosen = self.chosen_door.unwrap_or(0);
        let revealed = self.revealed_door.unwrap_or(0);
        self.winning_door * 9 + chosen * 3 + revealed
    }

    fn reset(&mut self) {
        let mut rng = rand::thread_rng();
        self.winning_door = rng.gen_range(0..3);
        self.chosen_door = None;
        self.revealed_door = None;
        self.final_choice = None;
    }

    fn is_game_over(&self) -> bool {
        self.final_choice.is_some()
    }

    fn available_actions(&self) -> Vec<usize> {
        match (self.chosen_door, self.revealed_door) {
            (None, _) => vec![0, 1, 2], // Premier choix de porte
            (Some(_), None) => vec![], // Révélation de la porte (automatique)
            (Some(_), Some(_)) => vec![0, 1], // Garder (0) ou Changer (1)
        }
    }

    fn score(&self) -> f32 {
        if let Some(final_choice) = self.final_choice {
            if final_choice == self.winning_door {
                1.0 // Gagné
            } else {
                0.0 // Perdu
            }
        } else {
            0.0 // Partie en cours
        }
    }

    fn step(&mut self, action: usize) {
        if self.is_game_over() {
            println!("Partie terminée. Aucun mouvement possible.");
            return;
        }

        if self.chosen_door.is_none() {
            self.chosen_door = Some(action);
            let mut rng = rand::thread_rng();
            let mut doors = vec![0, 1, 2];
            doors.retain(|&d| d != action && d != self.winning_door);
            self.revealed_door = Some(doors[rng.gen_range(0..doors.len())]);
        } else {
            if action == 1 {
                self.final_choice = Some(3 - self.chosen_door.unwrap() - self.revealed_door.unwrap());
            } else {
                self.final_choice = self.chosen_door;
            }
        }
    }

    fn display(&self) {
        println!("Winning door: {}", self.winning_door);
        println!("Chosen door: {:?}", self.chosen_door);
        println!("Revealed door: {:?}", self.revealed_door);
        println!("Final choice: {:?}", self.final_choice);
    }

    fn transition_probabilities(&self) -> Vec<Vec<Vec<f32>>> {
        vec![vec![vec![1.0; self.num_states()]; self.num_actions()]; self.num_states()]
    }

    fn reward_function(&self) -> Vec<Vec<f32>> {
        let mut rewards = vec![vec![0.0; self.num_actions()]; self.num_states()];
        for state in 0..self.num_states() {
            for action in 0..self.num_actions() {

                if self.is_game_over() {
                    rewards[state][action] = if self.final_choice == Some(self.winning_door) { 1.0 } else { 0.0 };
                }
            }
        }
        rewards
    }

    fn run_policy(&mut self, policy: &[usize]) -> f32 {
        let mut total_reward = 0.0;
        let mut switch_count = 0;
        let mut prev_choice = None;

        self.reset();

        for &action in policy {
            if self.is_game_over() {
                break;
            }

            prev_choice = self.chosen_door;
            self.step(action);


            println!("Action: {}, Choix initial: {:?}, Choix final: {:?}",
                     action, prev_choice, self.final_choice);

            if let (Some(prev), Some(final_choice)) = (prev_choice, self.final_choice) {
                if prev != final_choice {
                    switch_count += 1;
                    println!("L'agent a changé de porte!");
                }
            }

            let reward = self.score();
            println!("Récompense après l'action: {}", reward);
            total_reward += reward;
        }

        println!("L'agent a changé de porte {} fois.", switch_count);
        println!("Récompense totale: {}", total_reward);
        total_reward
    }
}
