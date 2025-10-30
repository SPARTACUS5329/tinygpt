use std::env;

use crate::{
    attention::generate_seq_matrix, embedder::embed, transformer::Transformer, utils::NiceError,
};

mod attention;
mod embedder;
mod tokenizer;
mod transformer;
mod utils;

fn main() -> Result<(), NiceError> {
    let args: Vec<String> = env::args().collect();
    let dataset_location = format!("./assets/{}", &args[1]);

    let seq_len = 32;
    let dim = 8;
    let eps = 0.003f32;
    let gamma: Vec<f32> = vec![1.0f32; dim as usize];
    let beta: Vec<f32> = vec![0.0f32; dim as usize];

    let content = utils::read_file(&dataset_location.to_string())?;
    let (tokens, dll_head) = tokenizer::tokenizer(content);
    let dll_head = dll_head.unwrap();

    println!("Tokenized the data with {} tokens", tokens.len());

    let (_token_to_vec, _vec_to_token) = embed(tokens, &dll_head);

    println!("Embedded the tokens into vectors of f32");

    let mut seq_matrix = generate_seq_matrix(seq_len, dim, &dll_head);
    let mut norm_seq = seq_matrix.layer_norm(eps, &gamma, &beta);

    let num_transformers = 4;
    let mut transformers: Vec<Transformer> = (0..num_transformers)
        .map(|_| Transformer::new(dim, seq_len))
        .collect();

    for transformer in transformers.iter_mut() {
        seq_matrix = transformer.run(&norm_seq);
        norm_seq = seq_matrix.layer_norm(eps, &gamma, &beta);
    }

    println!("{}", norm_seq);

    Ok(())
}
