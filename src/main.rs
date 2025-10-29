use std::env;

use crate::{attention::attention, embedder::embed, utils::NiceError};

mod attention;
mod embedder;
mod tokenizer;
mod utils;

fn main() -> Result<(), NiceError> {
    let args: Vec<String> = env::args().collect();
    let dataset_location = format!("./assets/{}", &args[1]);

    let content = utils::read_file(&dataset_location.to_string())?;
    let (tokens, dll_head) = tokenizer::tokenizer(content);
    let dll_head = dll_head.unwrap();

    println!("Tokenized the data with {} tokens", tokens.len());

    let (_token_to_vec, _vec_to_token) = embed(tokens, &dll_head);

    println!("Embedded the tokens into vectors of f32");

    let seq_len = 32;
    let dim = 8;

    let (_attention_params, output) = attention(seq_len, dim, &dll_head);
    println!("{}", output);

    Ok(())
}
