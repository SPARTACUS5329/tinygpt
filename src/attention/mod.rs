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
    pub fn new(dim: usize) -> Self {
        Self {
            dim: dim as i32,
            w_q: MatrixF32::new_rand_weight(dim, dim),
            w_k: MatrixF32::new_rand_weight(dim, dim),
            w_v: MatrixF32::new_rand_weight(dim, dim),
            w_o: MatrixF32::new_rand_weight(dim, dim),
        }
    }
}

pub fn generate_seq_matrix(
    seq_len: i32,
    dim: i32,
    seq_dll_head: &Rc<RefCell<DLLToken>>,
) -> (MatrixF32, Option<Rc<RefCell<DLLToken>>>, Vec<i32>) {
    let mut seq = MatrixF32::new(seq_len, dim);
    let mut seq_vals: Vec<f32> = vec![];
    let mut target_token_ids: Vec<i32> = vec![];

    let mut seq_dll_node = Some(Rc::clone(seq_dll_head));
    let mut pos = 0i32;

    while let Some(seq_dll_token) = seq_dll_node.clone()
        && pos < seq_len
    {
        seq_vals.extend(seq_dll_token.borrow().embed.clone());
        seq_dll_node = seq_dll_token.borrow().next.clone();
        if let Some(dll_node) = seq_dll_node.clone() {
            target_token_ids.push(dll_node.borrow().token.borrow().id);
        }
        pos += 1;
    }

    seq.vals = seq_vals;
    (seq, seq_dll_node, target_token_ids)
}

pub fn attention(dim: i32, seq: &MatrixF32) -> (AttentionParams, MatrixF32) {
    let attention_params = AttentionParams::new(dim as usize);
    let q = seq * &attention_params.w_q; // L * D
    let mut k = seq * &attention_params.w_k; // L * D
    let v = seq * &attention_params.w_v; // L * D

    k.transpose(); // D * L

    let mut scores = &q * &k; // L * L
    scores = &scores / (dim as f32).sqrt();
    scores.casual_mask();
    scores.softmax_row();
    let mut output = &scores * &v;
    output = &output * &attention_params.w_o;
    output = &output + seq;

    (attention_params, output)
}
