use std::{collections::HashMap, env};

use crate::{
    tokenizer::Token,
    utils::{NiceError, print_tokens},
};

mod tokenizer;
mod utils;

fn main() -> Result<(), NiceError> {
    let args: Vec<String> = env::args().collect();
    let dataset_location = format!("./assets/{}", &args[1]);

    let content = utils::read_file(&dataset_location.to_string())?;
    let (tokens, _dll_head) = tokenizer::tokenizer(content);

    // print_tokens(&tokens);

    let dim = 8i32;
    let embeddings = utils::random_embedding(tokens.len(), dim as usize);
    let _token_to_vec: HashMap<Token, Vec<f32>> = tokens
        .clone()
        .into_iter()
        .zip(embeddings.into_iter())
        .map(|(token_ref, vec)| {
            let token = token_ref.0.borrow();
            (token.clone(), vec)
        })
        .collect();

    Ok(())
}
