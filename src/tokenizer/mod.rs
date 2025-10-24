use core::num;
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    hash::{Hash, Hasher},
    rc::Rc,
};

#[derive(Clone)]
pub struct Token {
    pub id: i32,
    pub val: String,
    occurrences: HashSet<DLLTokenRef>,
}

impl Token {
    fn new(id: i32, val: String) -> Self {
        Token {
            id,
            val,
            occurrences: HashSet::new(),
        }
    }
}

impl PartialEq for Token {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Token {}

impl Hash for Token {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

#[derive(Clone)]
pub struct TokenRef(pub Rc<RefCell<Token>>);

impl PartialEq for TokenRef {
    fn eq(&self, other: &Self) -> bool {
        self.0.borrow().id == other.0.borrow().id
    }
}

impl Eq for TokenRef {}

impl Hash for TokenRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.borrow().id.hash(state);
    }
}

#[derive(Clone)]
pub struct DLLToken {
    pub id: i32,
    pub token: Rc<RefCell<Token>>,
    pub next: Option<Rc<RefCell<DLLToken>>>,
    pub prev: Option<Rc<RefCell<DLLToken>>>,
}

impl PartialEq for DLLToken {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for DLLToken {}

impl Hash for DLLToken {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

#[derive(Clone)]
pub struct DLLTokenRef(Rc<RefCell<DLLToken>>);

impl PartialEq for DLLTokenRef {
    fn eq(&self, other: &Self) -> bool {
        self.0.borrow().id == other.0.borrow().id
    }
}

impl Eq for DLLTokenRef {}

impl Hash for DLLTokenRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.borrow().id.hash(state);
    }
}

impl DLLToken {
    fn new(id: i32, token: Rc<RefCell<Token>>, prev: Option<Rc<RefCell<DLLToken>>>) -> Self {
        DLLToken {
            id: id,
            token,
            next: None,
            prev,
        }
    }
}

fn count_token_pairs(
    head: Rc<RefCell<DLLToken>>,
    token_pair_count_map: &mut HashMap<(String, String), i32>,
) -> Option<(Rc<RefCell<Token>>, Rc<RefCell<Token>>)> {
    let mut node = Some(head);
    let mut max_count_token_pair: Option<(Rc<RefCell<Token>>, Rc<RefCell<Token>>)> = None;
    let mut max_count = 0i32;

    while let Some(token) = node {
        let next_node = token.borrow().next.clone();

        if let Some(next) = next_node {
            let curr_val = token.borrow().token.borrow().val.clone();
            let next_val = next.borrow().token.borrow().val.clone();
            let pair = (curr_val, next_val);

            *token_pair_count_map.entry(pair.clone()).or_insert(0) += 1;
            let curr_count = token_pair_count_map.get(&pair).unwrap();
            if *curr_count > max_count {
                max_count_token_pair = Some((
                    Rc::clone(&token.borrow().token),
                    Rc::clone(&next.borrow().token),
                ));

                max_count = *curr_count;
            }

            node = Some(next);
        } else {
            break;
        }
    }

    max_count_token_pair
}

fn merge_tokens(
    token_a: Rc<RefCell<Token>>,
    token_b: Rc<RefCell<Token>>,
    token_pair_count_map: &mut HashMap<(String, String), i32>,
    num_tokens: &mut i32,
    num_dll_tokens: &mut i32,
) -> Option<Rc<RefCell<DLLToken>>> {
    let token_a_val = &token_a.borrow().val;
    let token_b_val = &token_b.borrow().val;
    let merge_pair = (token_a_val.clone(), token_b_val.clone());
    let merged_token_val = format!("{}{}", token_a_val, token_b_val);

    let merged_token = Rc::new(RefCell::new(Token::new(
        *num_tokens,
        merged_token_val.clone(),
    )));
    *num_tokens += 1;

    let mut new_head: Option<Rc<RefCell<DLLToken>>> = None;
    for dll_token in token_a.borrow().occurrences.iter() {
        let borrowed_dll_token = dll_token.0.borrow();
        if borrowed_dll_token.next.is_none() {
            continue;
        }

        let next_dll_token = borrowed_dll_token.next.as_ref().unwrap();
        if next_dll_token.borrow().token.borrow().id != token_b.borrow().id {
            continue;
        }

        let merged_dll_token = Rc::new(RefCell::new(DLLToken::new(
            *num_dll_tokens,
            Rc::clone(&merged_token),
            None,
        )));

        merged_token
            .borrow_mut()
            .occurrences
            .insert(DLLTokenRef(Rc::clone(&merged_dll_token)));

        *num_dll_tokens += 1;

        // Reduce the count of (a, b)
        token_pair_count_map
            .entry(merge_pair.clone())
            .and_modify(|count| *count -= 1);

        // Reduce the count of (prev_a, a)
        if let Some(prev) = &dll_token.0.borrow().prev {
            let prev_val = prev.borrow().token.borrow().val.clone();
            token_pair_count_map
                .entry((prev_val.clone(), token_a_val.clone()))
                .and_modify(|count| *count -= 1);
            token_pair_count_map
                .entry((prev_val.clone(), merged_token_val.clone()))
                .and_modify(|count| *count += 1)
                .or_insert(1);
            merged_dll_token.borrow_mut().prev = Some(Rc::clone(&prev));
            prev.borrow_mut().next = Some(Rc::clone(&merged_dll_token));
        } else {
            // If the first pair got merged
            new_head = Some(Rc::clone(&merged_dll_token));
        }

        // Reduce the count of (b, next_b)
        if let Some(next) = &next_dll_token.borrow().next {
            let next_val = next.borrow().token.borrow().val.clone();
            token_pair_count_map
                .entry((token_b_val.clone(), next_val.clone()))
                .and_modify(|count| *count -= 1);
            token_pair_count_map
                .entry((merged_token_val.clone(), next_val.clone()))
                .and_modify(|count| *count += 1)
                .or_insert(1);
            merged_dll_token.borrow_mut().next = Some(Rc::clone(&next));
            next.borrow_mut().prev = Some(Rc::clone(&merged_dll_token));
        }
    }

    return new_head;
}

fn init_tokens(
    sequence: String,
    token_map: &mut HashMap<String, Rc<RefCell<Token>>>,
    num_tokens: &mut i32,
    tokens: &mut HashSet<TokenRef>,
) -> (i32, Option<Rc<RefCell<DLLToken>>>) {
    let mut num_dll_tokens = 0i32;
    let mut prev: Option<Rc<RefCell<DLLToken>>> = None;
    let mut head: Option<Rc<RefCell<DLLToken>>> = None;

    for c in sequence.chars() {
        let token = token_map.entry(c.to_string()).or_insert_with(|| {
            let token = Rc::new(RefCell::new(Token::new(*num_tokens, c.to_string())));
            tokens.insert(TokenRef(Rc::clone(&token)));
            *num_tokens += 1;
            token
        });

        let dll_token = if let Some(prev_token) = &prev {
            let new_dll_token = Rc::new(RefCell::new(DLLToken::new(
                num_dll_tokens,
                Rc::clone(&token),
                Some(Rc::clone(prev_token)),
            )));

            prev_token.borrow_mut().next = Some(Rc::clone(&new_dll_token));
            Rc::clone(&new_dll_token)
        } else {
            head = Some(Rc::new(RefCell::new(DLLToken::new(
                num_dll_tokens,
                Rc::clone(&token),
                None,
            ))));
            head.clone().unwrap()
        };

        num_dll_tokens += 1;

        token
            .borrow_mut()
            .occurrences
            .insert(DLLTokenRef(Rc::clone(&dll_token)));

        prev = Some(Rc::clone(&dll_token));
    }

    return (num_dll_tokens, head);
}

pub fn tokenizer(sequence: String) -> (HashSet<TokenRef>, Option<Rc<RefCell<DLLToken>>>) {
    let mut tokens: HashSet<TokenRef> = HashSet::new();
    let mut token_map: HashMap<String, Rc<RefCell<Token>>> = HashMap::new();
    let mut num_tokens = 0i32;
    let mut token_pair_count_map: HashMap<(String, String), i32> = HashMap::new();

    let (mut num_dll_tokens, head) =
        init_tokens(sequence, &mut token_map, &mut num_tokens, &mut tokens);

    if head.is_none() {
        eprintln!("[tokenizer] No head found");
        return (tokens, None);
    }

    let mut dll_head = head.unwrap();

    let max_iters = 500;
    let mut iters = 0i32;

    while iters < max_iters {
        let max_pair = count_token_pairs(Rc::clone(&dll_head), &mut token_pair_count_map);
        if max_pair.is_none() {
            eprintln!("[tokenizer] max_pair not found");
            break;
        }

        let (token_a, token_b) = max_pair.unwrap();
        if let Some(new_head) = merge_tokens(
            token_a,
            token_b,
            &mut token_pair_count_map,
            &mut num_tokens,
            &mut num_dll_tokens,
        ) {
            dll_head = Rc::clone(&new_head);
        }

        iters += 1;
    }

    let mut final_tokens: HashSet<TokenRef> = HashSet::new();
    let mut dll_node = Some(Rc::clone(&dll_head));

    while let Some(node_rc) = dll_node {
        let node = node_rc.borrow();
        final_tokens.insert(TokenRef(Rc::clone(&node.token)));
        dll_node = node.next.clone();
    }

    return (final_tokens, Some(dll_head));
}
