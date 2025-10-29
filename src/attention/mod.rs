use crate::{tokenizer::DLLToken, utils::MatrixF32};
use std::{cell::RefCell, rc::Rc};

pub struct AttentionParams {
    dim: i32,
    w_q: MatrixF32,
    w_k: MatrixF32,
    w_v: MatrixF32,
    w_o: MatrixF32,
}

impl AttentionParams {
    fn new(dim: usize) -> Self {
        Self {
            dim: dim as i32,
            w_q: MatrixF32::new_rand_weight(dim, dim),
            w_k: MatrixF32::new_rand_weight(dim, dim),
            w_v: MatrixF32::new_rand_weight(dim, dim),
            w_o: MatrixF32::new_rand_weight(dim, dim),
        }
    }
}

pub fn attention(
    seq_len: i32,
    dim: i32,
    seq_dll_head: &Rc<RefCell<DLLToken>>,
) -> (AttentionParams, MatrixF32) {
    let attention_params = AttentionParams::new(dim as usize);
    let mut seq = MatrixF32::new(seq_len, dim);
    let mut seq_vals: Vec<f32> = vec![];

    let mut seq_dll_node = Some(Rc::clone(seq_dll_head));
    let mut pos = 0i32;

    while let Some(seq_dll_token) = seq_dll_node
        && pos < seq_len
    {
        seq_vals.extend(seq_dll_token.borrow().embed.clone());
        seq_dll_node = seq_dll_token.borrow().next.clone();
        pos += 1;
    }

    seq.vals = seq_vals;

    let q = &seq * &attention_params.w_q; // L * D
    let mut k = &seq * &attention_params.w_k; // L * D
    let v = &seq * &attention_params.w_v; // L * D

    k.transpose(); // D * L

    let mut scores = &q * &k; // L * L
    scores = &scores / (dim as f32).sqrt();
    scores.casual_mask();
    scores.softmax_row();
    let output = &(&scores * &v) * &attention_params.w_o;

    (attention_params, output)
}
