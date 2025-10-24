use std::env;

use crate::utils::NiceError;

mod tokenizer;
mod utils;

fn main() -> Result<(), NiceError> {
    let args: Vec<String> = env::args().collect();
    let dataset_location = format!("./assets/{}", &args[1]);

    let content = utils::read_file(&dataset_location.to_string())?;
    let (tokens, dll_head) = tokenizer::tokenizer(content);

    for token in tokens.iter() {
        println!("{}", token.0.borrow().val);
    }

    let mut dll_node = dll_head.clone();
    while let Some(dll_token_rc) = dll_node {
        let dll_token = dll_token_rc.borrow();
        println!(
            "{} -> {} {}",
            dll_token.token.borrow().val,
            dll_token.id,
            dll_token.token.borrow().id
        );
        dll_node = dll_token.next.clone();
    }

    Ok(())
}
