use crate::Environment;


#[derive(Clone)]
pub struct LineWorld {
    state: usize,
    // Ajoute d'autres champs si nécessaire pour LineWorld
}

impl LineWorld {
    // Constructeur et autres méthodes déjà définis
    pub fn new() -> Self {
        LineWorld { state: 2 } // Par exemple, état initial au milieu
    }
}

impl Environment for LineWorld {
    fn new() -> Self {
        LineWorld::new()
    }

    fn num_states(&self) -> usize {
        5 // Exemple: 5 états dans le LineWorld
    }

    fn num_actions(&self) -> usize {
        2 // Exemple: 2 actions possibles (aller à gauche, aller à droite)
    }

    fn state_id(&self) -> usize {
        self.state
    }

    fn reset(&mut self) {
        self.state = 2; // Remise à l'état initial
    }

    fn is_game_over(&self) -> bool {
        self.state == 0 || self.state == 4 // Par exemple, le jeu est fini si on atteint un des bords
    }

    fn available_actions(&self) -> Vec<usize> {
        if self.state == 0 {
            vec![1] // On peut seulement aller à droite
        } else if self.state == 4 {
            vec![0] // On peut seulement aller à gauche
        } else {
            vec![0, 1] // On peut aller à gauche ou à droite
        }
    }

    fn score(&self) -> f32 {
        if self.is_game_over() {
            1.0
        } else {
            0.0
        }
    }

    fn step(&mut self, action: usize) {
        if action == 0 && self.state > 0 {
            self.state -= 1; // Aller à gauche
        } else if action == 1 && self.state < 4 {
            self.state += 1; // Aller à droite
        }
    }

    fn display(&self) {
        // Affichage de l'état actuel de l'environnement
        println!("Current state: {}", self.state);
    }

    fn transition_probabilities(&self) -> Vec<Vec<Vec<f32>>> {
        let mut prob = vec![vec![vec![0.0; self.num_states()]; self.num_actions()]; self.num_states()];

        for state in 0..self.num_states() {
            for action in 0..self.num_actions() {
                if action == 0 { // Action "aller à gauche"
                    if state > 0 {
                        prob[state][action][state - 1] = 1.0;
                    } else {
                        prob[state][action][state] = 1.0;
                    }
                } else if action == 1 { // Action "aller à droite"
                    if state < self.num_states() - 1 {
                        prob[state][action][state + 1] = 1.0;
                    } else {
                        prob[state][action][state] = 1.0;
                    }
                }
            }
        }

        prob
    }

    fn reward_function(&self) -> Vec<Vec<f32>> {
        let mut rewards = vec![vec![0.0; self.num_actions()]; self.num_states()];

        for state in 0..self.num_states() {
            for action in 0..self.num_actions() {
                if self.is_game_over() {
                    rewards[state][action] = 1.0;
                } else {
                    rewards[state][action] = 0.0;
                }
            }
        }

        rewards
    }

    // Implémentation de la méthode run_policy
    fn run_policy(&mut self, policy: &[usize]) -> f32 {
        let mut total_reward = 0.0;
        let mut steps = 0;

        self.reset(); // Commence à l'état initial

        while !self.is_game_over() && steps < 1000 {
            let action = policy[self.state_id()]; // Prendre l'action selon la politique
            self.step(action);
            total_reward += self.score(); // Ajouter la récompense de l'état actuel
            steps += 1;
        }

        total_reward / steps as f32 // Retourner la récompense moyenne
    }
}


