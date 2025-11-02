use std::{env, rc::Rc};

use crate::{embedder::embed, model::Model, tokenizer::Token, utils::NiceError};

mod attention;
mod embedder;
mod model;
mod tokenizer;
mod transformer;
mod utils;

fn main() -> Result<(), NiceError> {
    let args: Vec<String> = env::args().collect();
    let dataset_location = format!("./assets/{}", &args[1]);

    let content = utils::read_file(&dataset_location.to_string())?;
    let (tokens, dll_head, token_id_map) = tokenizer::tokenizer(content);
    let token_list: Vec<Token> = tokens.iter().map(|t| t.0.borrow().clone()).collect();

    let dll_head = dll_head.unwrap();

    let vocab_size = tokens.len();
    let seq_len = 32;
    let dim = 8;
    let eps = 0.003f32;

    let model = Model::new(seq_len, eps, dim, vocab_size as i32, token_id_map);

    println!("Tokenized the data with {} tokens", tokens.len());
    let (_token_to_vec, _vec_to_token) = embed(tokens, &dll_head);

    println!("Embedded the tokens into vectors of f32");

    for _ in 0..4 {
        let (vocab_pred, next_dll_head, target_token_ids) = model.forward(Rc::clone(&dll_head));
        let loss = model.cross_entropy(vocab_pred, target_token_ids);
        println!("{}", loss);
    }

    // println!("{}", norm_seq);

    Ok(())
}
