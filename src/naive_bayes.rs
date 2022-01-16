use std::vec::Vec;
use std::collections::{HashSet,HashMap};
use regex::Regex;

pub struct Model{
    vocab: HashSet<String>,
    classifications: HashMap<String, Classification>,
    num_examples: u32,
    smoothing: f64,
}
struct Classification {
    label: String,
    num_examples: u32,
    num_words: u32,
    probability: f64,
    zero_word_probability: f64,
    words_dist: HashMap<String, (u32, f64)>, // word counter and prob
}

impl Classification {
    fn new(label: &String) -> Classification {
        Classification {
            label: label.clone(),
            num_examples: 0_u32,
            num_words: 0_u32,
            probability: 0.0_f64,
            zero_word_probability: 0.0_f64,
            words_dist: HashMap::new(),
        }
    }
    pub fn count_word(&mut self, word:&String){
        // count word 
        // returns a mutable reference
        let mut word_dist = self.words_dist.entry(word.to_string()).or_insert((1, 0.0f64));
        //counting
        word_dist.0+=1
    }
    pub fn train(&mut self ,vocab: &HashSet<String> ,total_examples:u32, smoothing:f64){
        // calculate p(w|c) / p(c)
        self.probability = self.num_examples as f64 / total_examples as f64;
        self.zero_word_probability = smoothing / 
            (self.num_words as f64 + smoothing * vocab.len() as f64);
        for word in vocab.iter(){
            if self.words_dist.contains_key(word) {
                let word_entry = self.words_dist.get_mut(word).expect("no word exist");
                let word_count = word_entry.0;
                let p_word_given_label =
                    (word_count as f64 + smoothing) /
                    (self.num_words as f64 + smoothing * vocab.len() as f64);
                // update word distribution
                word_entry.1 = p_word_given_label;
            }
        }
    }
    pub fn score_document(& self,token:&Vec<String>) -> f64 {
        let mut score = 0.0_f64;
        for word in token.iter(){
            let prob  = match  self.words_dist.get(word){
                Some(&(_,p )) => p,
                None => self.zero_word_probability,
            };
            score += prob.ln();
        }
        // log p(c) + log p(w|c) 
        score += self.probability.ln();
        return score;
    }
}

impl Model {
    pub fn new() -> Model{
        Model{
            vocab: HashSet::new(),
            classifications: HashMap::new(),
            num_examples:0,
            smoothing: 1.0_f64,
        }
    }
    pub fn add_document(&mut self,document:&String,label:&String) {
        // tokenzie with whitespace
        // TODO:: other tokenized method
        self.add_document_tokenized(&split_document(document), label);
    }
    pub fn add_document_tokenized(&mut self,tokens:&Vec<String>,label:&String) {
        if tokens.len() == 0 {return ;}
        //let mut classification = self.classifications.get_mut(label).unwrap();
        let c_label =self.classifications.entry(label.to_string()).or_insert(Classification::new(label));
        for tk in tokens {
            c_label.count_word(tk);
            self.vocab.insert(tk.to_string());
        }
        self.num_examples += 1;
        c_label.num_examples +=1
    }
    pub fn train(&mut self){
        for (_, classification) in self.classifications.iter_mut() {
            classification.train(&self.vocab, self.num_examples, self.smoothing);
            //println!("prob_{} {:?}",classification.label,classification.probability);
        }
    }
    pub fn classify_tokenized(& mut self, token:&Vec<String> ) -> String{
        let mut prediction ="".to_string();
        let mut max_score = f64::NEG_INFINITY;
        for c in self.classifications.values(){
            let score = c.score_document(token);
            if score > max_score {
                max_score  = score; 
                prediction = c.label.clone()
            }
        }
        //println!("prediction : {} score:{}",prediction,max_score);
        return prediction;
    }
    pub fn classify(& mut self, document:&String) -> String {
        self.classify_tokenized(&split_document(document))
    }
    pub fn predict(& mut self, target:&Vec<String>) -> Vec<String> {
        let mut prediction = Vec::new();
        for rec in target{
            prediction.push(self.classify(rec));
        }
        prediction
    }

}

// splits a String on whitespaces
fn split_document(document: &String) -> Vec<String> {
    let re = Regex::new(r"(\s)").unwrap();
    re.split(document).map(|s| s.to_string()).collect()
}
