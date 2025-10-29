use core::{f32, fmt};
use std::collections::HashSet;
use std::fs::OpenOptions;
use std::io::Read;
use std::ops::{Div, Mul};
use std::ops::{Index, IndexMut};

use rand::thread_rng;
use rand_distr::uniform::SampleBorrow;
use rand_distr::{Distribution, Uniform};

use crate::tokenizer::TokenRef;

#[derive(Debug)]
pub struct NiceError {
    message: String,
}

impl NiceError {
    pub fn new(message: String) -> NiceError {
        eprintln!("{}", message);
        let error = NiceError { message };
        error
    }

    pub fn show(self) -> NiceError {
        eprintln!("{}", self.message);
        self
    }
}

pub struct MatrixF32 {
    pub rows: i32,
    pub cols: i32,
    pub vals: Vec<f32>,
}

impl fmt::Display for MatrixF32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Iterate over rows
        for r in 0..self.rows {
            let start = (r * self.cols) as usize;
            let end = start + self.cols as usize;
            let row_vals = &self.vals[start..end];

            // Join values as a space-separated string
            let row_str = row_vals
                .iter()
                .map(|v| format!("{:.3}", v)) // format each value to 3 decimal places
                .collect::<Vec<String>>()
                .join(" ");

            writeln!(f, "{}", row_str)?; // write row to formatter
        }
        Ok(())
    }
}

impl MatrixF32 {
    pub fn new(rows: i32, cols: i32) -> Self {
        Self {
            rows,
            cols,
            vals: vec![0.0; (rows * cols) as usize],
        }
    }

    pub fn new_rand_weight(rows: usize, cols: usize) -> Self {
        let limit = (6.0 / (rows as f32 + cols as f32)).sqrt();
        let uniform = Uniform::new(-limit, limit);
        let mut rng = thread_rng();

        let vals = (0..rows)
            .map(|_| {
                (0..cols)
                    .map(|_| uniform.sample(&mut rng))
                    .collect::<Vec<f32>>()
            })
            .flatten()
            .collect();

        Self {
            rows: rows as i32,
            cols: cols as i32,
            vals,
        }
    }

    pub fn transpose(&mut self) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                let prev_ij = self[(i, j)];
                self[(i, j)] = self[(j, i)];
                self[(j, i)] = prev_ij;
            }
        }
        let prev_rows = self.rows;
        self.rows = self.cols;
        self.cols = prev_rows;
    }

    pub fn casual_mask(&mut self) {
        for i in 0..self.rows {
            for j in (i + 1)..self.cols {
                self[(i, j)] = f32::NEG_INFINITY;
            }
        }
    }

    pub fn softmax_row(&mut self) {
        for i in 0..self.rows {
            let mut max = f32::NEG_INFINITY;
            for j in 0..self.cols {
                max = f32::max(max, self[(i, j)]);
            }

            let mut exp_sum = 0.0;
            for j in 0..self.cols {
                exp_sum += (self[(i, j)] - max).exp()
            }

            for j in 0..self.cols {
                self[(i, j)] = (self[(i, j)] - max).exp() / exp_sum;
            }
        }
    }
}

impl Index<(i32, i32)> for MatrixF32 {
    type Output = f32;

    fn index(&self, (i, j): (i32, i32)) -> &Self::Output {
        let index = i * self.cols + j;
        &self.vals[index as usize]
    }
}

impl IndexMut<(i32, i32)> for MatrixF32 {
    fn index_mut(&mut self, (i, j): (i32, i32)) -> &mut Self::Output {
        let index = i * self.cols + j;
        &mut self.vals[index as usize]
    }
}

impl Mul for &MatrixF32 {
    type Output = MatrixF32;

    fn mul(self, other: &MatrixF32) -> MatrixF32 {
        let c_rows = self.rows;
        let c_cols = other.cols;
        let mut c = MatrixF32::new(c_rows, c_cols);

        let a_rows = self.rows;
        let a_cols = self.cols;
        let b_cols = other.cols;

        for a_row in 0..a_rows {
            for b_col in 0..b_cols {
                for a_col in 0..a_cols {
                    c[(a_row, b_col)] += self[(a_row, a_col)] * other[(a_col, b_col)];
                }
            }
        }

        c
    }
}

impl<'a> Div<f32> for &'a MatrixF32 {
    type Output = MatrixF32;

    fn div(self, rhs: f32) -> MatrixF32 {
        if rhs == 0.0 {
            panic!("Division by zero");
        }

        let vals = self.vals.iter().map(|v| v / rhs).collect();

        MatrixF32 {
            rows: self.rows,
            cols: self.cols,
            vals,
        }
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
