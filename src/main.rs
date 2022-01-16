use naiveb_rust::NB;
extern crate csv;
use serde::Deserialize;
mod data_knife;




#[derive(Debug,Clone, Deserialize)]
struct Data{
    title: String,
    text: String,
    subject: String,
    date: String,
    #[serde(default)]
    label: String 
}


fn main() {
    let file_path = "./data/Fake.csv";
    let mut rdr = csv::Reader::from_path(file_path).unwrap();
    let mut x = Vec::new();
    let mut y = Vec::new();
    for r in rdr.deserialize() {
        let mut record: Data =r.unwrap();
        record.label = "fake".to_string();
        x.push(record.text.clone());
        y.push(record.label.clone());
    }
    let file_path = "./data/True.csv";
    let mut rdr = csv::Reader::from_path(file_path).unwrap();
    for r in rdr.deserialize() {
        let mut record: Data =r.unwrap();
        record.label = "true".to_string();
        x.push(record.text.clone());
        y.push(record.label.clone());
    }

    let  (x_train, x_test, y_train, y_test) = data_knife::train_test_split(&x,&y,0.7,true);

    // create a new classifier
    let mut nb = NB::new();


    for i in 0..x_train.len() {
        nb.add_document(&x_train[i].to_string(), &y_train[i].to_string());
    }
    
    // train the classifier
    nb.train();
    let y_predict = nb.predict(&x_test);
    println!("score:{}",data_knife::calculate_score(&y_test,&y_predict))
    
}
