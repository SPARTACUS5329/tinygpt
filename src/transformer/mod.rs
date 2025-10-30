use crate::{
    attention::{AttentionParams, attention},
    utils::{MatrixF32, NeuralNetwork},
};

pub struct Transformer {
    pub dim: i32,
    pub seq_len: i32,
    pub attention_eps: f32,
    pub attention_beta: Vec<f32>,
    pub attention_gamma: Vec<f32>,
    pub ff_eps: f32,
    pub ff_beta: Vec<f32>,
    pub ff_gamma: Vec<f32>,
    pub attention_params: AttentionParams,
    pub nn: NeuralNetwork,
}

impl Transformer {
    pub fn new(dim: i32, seq_len: i32) -> Self {
        let nn_hidden_nodes = 32;
        let mut nn = NeuralNetwork::new(seq_len, dim);
        nn.add_layer(nn_hidden_nodes, dim);
        nn.add_layer(dim, nn_hidden_nodes);

        Self {
            dim,
            seq_len,
            attention_params: AttentionParams::new(dim as usize),
            nn,
            attention_eps: 0.003f32,
            attention_gamma: vec![1.0f32; dim as usize],
            attention_beta: vec![0.0f32; dim as usize],
            ff_eps: 0.003f32,
            ff_gamma: vec![1.0f32; dim as usize],
            ff_beta: vec![0.0f32; dim as usize],
        }
    }

    pub fn run(self: &mut Transformer, seq: &MatrixF32) -> MatrixF32 {
        let (_attention_params, mut output) = attention(self.dim, &seq);
        output = output.layer_norm(
            self.attention_eps,
            &self.attention_gamma,
            &self.attention_beta,
        );

        let mut nn_output = self.nn.feed_forward(&output);
        nn_output = &nn_output + &output;
        nn_output.layer_norm(self.ff_eps, &self.ff_gamma, &self.ff_beta);
        nn_output
    }
}
