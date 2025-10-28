use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    rc::Rc,
};

use rand_distr::{Distribution, Uniform, uniform::SampleBorrow};

use crate::tokenizer::{DLLToken, Token, TokenRef};

pub fn random_embedding(vocab_size: usize, dim: usize) -> Vec<Vec<f32>> {
    let range = Uniform::new(-0.1, 0.1);
    let mut rng = rand::thread_rng();
    (0..vocab_size)
        .map(|_| (0..dim).map(|_| range.sample(&mut rng)).collect())
        .collect()
}

pub fn quantize_f32(vec: Vec<f32>) -> Vec<i32> {
    let vec_key: Vec<i32> = vec.iter().map(|f| (1000.0 * f) as i32).collect();
    vec_key
}

fn positional_encoding(pos: i32, dim: i32) -> Vec<f32> {
    let mut enc = Vec::with_capacity(dim as usize);
    for i in 0..dim {
        let angle = pos as f32 / (10000_f32.powf((2.0 * ((i / 2) as f32)) as f32 / (dim as f32)));
        if i % 2 == 0 {
            enc.push(angle.sin());
        } else {
            enc.push(angle.cos());
        }
    }
    enc
}

pub fn embed(
    tokens: HashSet<TokenRef>,
    token_dll_head: &Rc<RefCell<DLLToken>>,
) -> (HashMap<Token, Vec<f32>>, HashMap<Vec<i32>, Token>) {
    let dim = 8i32;
    let embeddings = random_embedding(tokens.len(), dim as usize);

    let mut token_to_vec: HashMap<Token, Vec<f32>> = HashMap::new();
    let mut vec_to_token: HashMap<Vec<i32>, Token> = HashMap::new();

    for (token_ref, vec) in tokens.iter().zip(embeddings.into_iter()) {
        let token = token_ref.0.borrow().clone();
        token_to_vec.insert(token.clone(), vec.clone());
        let vec_key = quantize_f32(vec);
        vec_to_token.insert(vec_key, token.clone());
    }

    let mut pos = 0i32;
    let mut token_node = Some(Rc::clone(token_dll_head));

    while let Some(dll_token) = token_node {
        let pos_enc = positional_encoding(pos, dim);
        let token_embed = token_to_vec
            .get(&dll_token.borrow().token.borrow())
            .unwrap();

        dll_token.borrow_mut().embed = token_embed
            .iter()
            .zip(pos_enc)
            .map(|(t, p)| t.borrow() + p)
            .collect();

        token_node = dll_token.borrow().next.clone();
        pos += 1;
    }

    (token_to_vec, vec_to_token)
}
