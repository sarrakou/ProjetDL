use std::ffi::c_void;
use std::sync::Arc;
use libloading::{Library, Symbol};
use crate::Environment;

pub struct SecretEnv {
    env_ptr: *mut c_void,
    lib: Arc<Library>,
    env_id: usize,
}

unsafe impl Send for SecretEnv {}
unsafe impl Sync for SecretEnv {}

impl Clone for SecretEnv {
    fn clone(&self) -> Self {
        unsafe {
            // Create a new environment with the same library
            let new_fn: Symbol<unsafe extern fn() -> *mut c_void> =
                self.lib.get(format!("secret_env_{}_new", self.env_id).as_bytes())
                    .expect("Failed to load new function");

            let env_ptr = new_fn();

            SecretEnv {
                env_ptr,
                lib: Arc::clone(&self.lib),
                env_id: self.env_id,
            }
        }
    }
}

impl SecretEnv {
    pub fn new(env_id: usize) -> Self {
        unsafe {
            #[cfg(target_os = "linux")]
            let path = "./libs/libsecret_envs.so";
            #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
            let path = "./libs/libsecret_envs_intel_macos.dylib";
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            let path = "./libs/libsecret_envs.dylib";
            #[cfg(windows)]
            let path = "./libs/secret_envs.dll";

            let lib = Arc::new(Library::new(path).expect("Failed to load library"));

            let new_fn: Symbol<unsafe extern fn() -> *mut c_void> =
                lib.get(format!("secret_env_{}_new", env_id).as_bytes())
                    .expect("Failed to load new function");

            let env_ptr = new_fn();

            SecretEnv {
                env_ptr,
                lib,
                env_id,
            }
        }
    }
}

impl Drop for SecretEnv {
    fn drop(&mut self) {
        unsafe {
            if let Ok(delete_fn) = self.lib.get::<unsafe extern fn(*mut c_void)>(
                format!("secret_env_{}_delete", self.env_id).as_bytes()
            ) {
                delete_fn(self.env_ptr);
            }
        }
    }
}

impl Environment for SecretEnv {
    fn new() -> Self {
        Self::new(0)  // Default to environment 0
    }

    fn num_states(&self) -> usize {
        unsafe {
            let num_states_fn: Symbol<unsafe extern fn() -> usize> =
                self.lib.get(format!("secret_env_{}_num_states", self.env_id).as_bytes())
                    .expect("Failed to load num_states function");
            num_states_fn()
        }
    }

    fn num_actions(&self) -> usize {
        unsafe {
            let num_actions_fn: Symbol<unsafe extern fn() -> usize> =
                self.lib.get(format!("secret_env_{}_num_actions", self.env_id).as_bytes())
                    .expect("Failed to load num_actions function");
            num_actions_fn()
        }
    }

    fn state_id(&self) -> usize {
        unsafe {
            let state_id_fn: Symbol<unsafe extern fn(*const c_void) -> usize> =
                self.lib.get(format!("secret_env_{}_state_id", self.env_id).as_bytes())
                    .expect("Failed to load state_id function");
            state_id_fn(self.env_ptr)
        }
    }

    fn reset(&mut self) {
        unsafe {
            let reset_fn: Symbol<unsafe extern fn(*mut c_void)> =
                self.lib.get(format!("secret_env_{}_reset", self.env_id).as_bytes())
                    .expect("Failed to load reset function");
            reset_fn(self.env_ptr);
        }
    }

    fn is_game_over(&self) -> bool {
        unsafe {
            let is_game_over_fn: Symbol<unsafe extern fn(*const c_void) -> bool> =
                self.lib.get(format!("secret_env_{}_is_game_over", self.env_id).as_bytes())
                    .expect("Failed to load is_game_over function");
            is_game_over_fn(self.env_ptr)
        }
    }

    fn available_actions(&self) -> Vec<usize> {
        unsafe {
            let available_actions_fn: Symbol<unsafe extern fn(*const c_void) -> *const usize> =
                self.lib.get(format!("secret_env_{}_available_actions", self.env_id).as_bytes())
                    .expect("Failed to load available_actions function");

            let available_actions_len_fn: Symbol<unsafe extern fn(*const c_void) -> usize> =
                self.lib.get(format!("secret_env_{}_available_actions_len", self.env_id).as_bytes())
                    .expect("Failed to load available_actions_len function");

            let actions_ptr = available_actions_fn(self.env_ptr);
            let len = available_actions_len_fn(self.env_ptr);

            let mut actions = Vec::with_capacity(len);
            for i in 0..len {
                actions.push(*actions_ptr.add(i));
            }

            let delete_fn: Symbol<unsafe extern fn(*const usize, usize)> =
                self.lib.get(format!("secret_env_{}_available_actions_delete", self.env_id).as_bytes())
                    .expect("Failed to load available_actions_delete function");
            delete_fn(actions_ptr, len);

            actions
        }
    }

    fn score(&self) -> f32 {
        unsafe {
            let score_fn: Symbol<unsafe extern fn(*const c_void) -> f32> =
                self.lib.get(format!("secret_env_{}_score", self.env_id).as_bytes())
                    .expect("Failed to load score function");
            score_fn(self.env_ptr)
        }
    }

    fn step(&mut self, action: usize) {
        unsafe {
            let step_fn: Symbol<unsafe extern fn(*mut c_void, usize)> =
                self.lib.get(format!("secret_env_{}_step", self.env_id).as_bytes())
                    .expect("Failed to load step function");
            step_fn(self.env_ptr, action);
        }
    }

    fn display(&self) {
        unsafe {
            let display_fn: Symbol<unsafe extern fn(*const c_void)> =
                self.lib.get(format!("secret_env_{}_display", self.env_id).as_bytes())
                    .expect("Failed to load display function");
            display_fn(self.env_ptr);
        }
    }
}