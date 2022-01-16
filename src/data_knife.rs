use rand::seq::SliceRandom;
pub fn train_test_split<T: std::clone::Clone> (
    x: &Vec<T>,
    y: &Vec<T>,
    test_size: f32,
    shuffle: bool,
) -> (Vec<T>,Vec<T>,Vec<T>,Vec<T>) {
    if x.len() != y.len() {
        panic!(
            "x and y should have the same number of samples. |x|: {}, |y|: {}",
            x.len(),
            y.len()
        );
    }

    if test_size <= 0. || test_size > 1.0 {
        panic!("test_size should be between 0 and 1");
    }

    let n = y.len();

    let n_test = ((n as f32) * test_size) as usize;

    if n_test < 1 {
        panic!("number of sample is too small {}", n);
    }

    let mut x_train = Vec::new();
    let mut x_test  = Vec::new();
    let mut y_train = Vec::new();
    let mut y_test  = Vec::new();


    if shuffle {
        let mut indices: Vec<usize> = (0..n).collect();
        let mut rng = rand::thread_rng();
        indices.shuffle(&mut rng);
        for id in indices[0..n_test].to_vec(){
            x_test.push(x[id].clone());
            y_test.push(y[id].clone());
        }
        for id in indices[n_test..n].to_vec(){
            x_train.push(x[id].clone());
            y_train.push(y[id].clone());
        }
    }else{
        x_train = x[n_test..n].to_vec();
        x_test = x[0..n_test].to_vec();
        y_train = y[n_test..n].to_vec();
        y_test = x[0..n_test].to_vec();
    }


    (x_train, x_test, y_train, y_test)
}

pub fn calculate_score<T: std::cmp::PartialEq>(y_true: &Vec<T>, y_pred: &Vec<T>) -> f64 {
    if y_true.len() != y_pred.len() {
        panic!(
            "The vector sizes don't match: {} != {}",
            y_true.len(),
            y_pred.len()
        );
    }

    let n = y_true.len();

    let mut positive = 0;
    for i in 0..n {
        if y_true.get(i) == y_pred.get(i) {
            positive += 1;
        }
    }

    positive as f64 / n as f64
}

