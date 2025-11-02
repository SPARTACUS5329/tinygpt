use std::{cell::RefCell, rc::Rc, usize};

use crate::{
    attention::generate_seq_matrix, tokenizer::DLLToken, transformer::Transformer, utils::MatrixF32,
};

pub struct Model {
    pub seq_len: i32,
    pub dim: i32,
    pub eps: f32,
    pub gamma: Vec<f32>,
    pub beta: Vec<f32>,
    pub vocab_size: i32,
    pub w_o: MatrixF32,
    token_id_map: Vec<i32>,
}

impl Model {
    pub fn new(seq_len: i32, eps: f32, dim: i32, vocab_size: i32, token_id_map: Vec<i32>) -> Self {
        Self {
            seq_len,
            dim,
            eps,
            gamma: vec![1.0; dim as usize],
            beta: vec![0.0; dim as usize],
            vocab_size,
            w_o: MatrixF32::new_rand_weight(dim as usize, vocab_size as usize),
            token_id_map,
        }
    }

    pub fn forward(
        self: &Model,
        dll_head: Rc<RefCell<DLLToken>>,
    ) -> (MatrixF32, Option<Rc<RefCell<DLLToken>>>, Vec<i32>) {
        let (mut seq_matrix, next_seq_head, target_token_ids) =
            generate_seq_matrix(self.seq_len, self.dim, &dll_head);
        let mut norm_seq = seq_matrix.layer_norm(self.eps, &self.gamma, &self.beta);

        let num_transformers = 4;
        let mut transformers: Vec<Transformer> = (0..num_transformers)
            .map(|_| Transformer::new(self.dim, self.seq_len))
            .collect();

        for transformer in transformers.iter_mut() {
            seq_matrix = transformer.run(&norm_seq);
            norm_seq = seq_matrix.layer_norm(self.eps, &self.gamma, &self.beta);
        }

        norm_seq = &norm_seq * &self.w_o;
        norm_seq.softmax_row();
        (norm_seq, next_seq_head, target_token_ids)
    }

    pub fn cross_entropy(self: &Model, vocab_pred: MatrixF32, target: Vec<i32>) -> f32 {
        let mut loss = 0f32;

        for i in 0..self.seq_len {
            let target_token_id = target[i as usize];
            let target_vocab_index = self.token_id_map[target_token_id as usize];
            let pred = vocab_pred[(i, target_vocab_index)];
            loss += -pred.ln();
        }

        loss / self.seq_len as f32
    }
}
