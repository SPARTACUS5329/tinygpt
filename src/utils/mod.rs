use rand_distr::{Distribution, Uniform};
use std::collections::HashSet;
use std::fs::OpenOptions;
use std::io::Read;

use crate::tokenizer::TokenRef;

#[derive(Debug)]
pub struct NiceError {
    message: String,
}

impl NiceError {
    pub fn new(message: String) -> NiceError {
        let error = NiceError { message };
        error
    }

    pub fn show(self) -> NiceError {
        eprintln!("{}", self.message);
        self
    }
}

pub fn read_file(filename: &String) -> Result<String, NiceError> {
    let mut file = match OpenOptions::new().read(true).open(filename) {
        Ok(file) => file,
        Err(error) => {
            return Err(NiceError::new(format!("Error opening file: {:?}", error)));
        }
    };

    let mut contents = String::new();

    let _ = match file.read_to_string(&mut contents) {
        Ok(_) => Ok(()),
        Err(_) => Err(NiceError::new(format!(
            "Error opening file: {:?}",
            filename
        ))),
    };

    Ok(contents)
}

pub fn print_tokens(tokens: &HashSet<TokenRef>) {
    for token in tokens.iter() {
        println!("{}", token.0.borrow().val);
    }
}

pub fn random_embedding(vocab_size: usize, dim: usize) -> Vec<Vec<f32>> {
    let range = Uniform::new(-0.1, 0.1);
    let mut rng = rand::thread_rng();
    (0..vocab_size)
        .map(|_| (0..dim).map(|_| range.sample(&mut rng)).collect())
        .collect()
}
