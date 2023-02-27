#![allow(unused)]
mod file;

use std::ops::{Mul, Add, Neg, Div, Sub, AddAssign, MulAssign};

use nalgebra::{DVector, DMatrix, Scalar, DVectorView, dvector};
use num_traits::Float;
use rand::prelude::*;
use rand_distr::{Normal, StandardNormal};

use file::ReadExt;

use crate::file::read_idx_path;

pub trait NetworkValue: Float + Scalar + AddAssign + MulAssign {}
impl<T> NetworkValue for T where T: Float + Scalar + AddAssign + MulAssign {}

pub trait ActivationFn<T> {
    fn calc(&self, x: T) -> T;
    fn calc_derivative(&self, x: T) -> T;
}

pub trait CostFn<T> {
    fn calc(&self, output: T, expected: T) -> T;
    fn calc_derivative(&self, output: T, expected: T) -> T;
}

pub trait Exp: Sized { fn exp(self) -> Self; }
impl Exp for f32 { fn exp(self) -> Self { self.exp() } }
impl Exp for f64 { fn exp(self) -> Self { self.exp() } }

#[derive(Default)]
pub struct Sigmoid;
impl<T: NetworkValue> ActivationFn<T> for Sigmoid {
    fn calc(&self, x: T) -> T { T::one() / (T::one() + (-x).exp()) }
    fn calc_derivative(&self, x: T) -> T { let a = self.calc(x); a * (T::one() - a) }
}

#[derive(Default)]
pub struct SquaredError;
impl<T: NetworkValue> CostFn<T> for SquaredError {
    fn calc(&self, output: T, expected: T) -> T { let v = output - expected; v * v }
    fn calc_derivative(&self, output: T, expected: T) -> T { (T::one() + T::one()) * (output - expected) }
}



pub struct Layer<T> {
    weights: DMatrix<T>,
}

impl<T: NetworkValue> Layer<T> {
    fn new(nodes_in: usize, nodes_out: usize) -> Self {
        let weights = DMatrix::zeros(nodes_out, nodes_in);
        Self { weights }
    }

    fn nodes_in(&self) -> usize { self.weights.ncols() }
    fn nodes_out(&self) -> usize { self.weights.nrows() }

    fn calculate(&self, input: &DVector<T>, activation: &impl ActivationFn<T>) -> DVector<T> {
        (&self.weights * input).map(|v| activation.calc(v))
    }
}

impl<T> Layer<T>
where
    T: NetworkValue,
    StandardNormal: Distribution<T>,
{
    fn new_random(nodes_in: usize, nodes_out: usize) -> Self {
        let mut rng = rand::thread_rng();
        let one = T::one();
        let mut dist: Normal<T> = Normal::new(
            T::zero(),
            T::from(nodes_out)
                .expect("invalid number of nodes_out")
            .powf(-(one / (one + one))),
        ).expect("could not create normal dist");

        let weights = DMatrix::from_fn(nodes_out, nodes_in, |_, _| { dist.sample(&mut rng) });
        Self { weights }
    }
}

pub struct Network<T, A, C> {
    layers: Vec<Layer<T>>,
    activation_fn: A,
    cost_fn: C,
}

impl<T, A, C> Network<T, A, C>
where
    T: NetworkValue,
    A: ActivationFn<T>,
    C: CostFn<T>,
{
    fn new_fn(layers: &[usize], activation_fn: A, cost_fn: C) -> Self {
        let layers = layers.windows(2)
            .map(|v| Layer::new(v[0], v[1]))
        .collect();

        Self { layers, activation_fn, cost_fn }
    }

    fn save(&self) { todo!() }
    fn load(&self) { todo!() }

    fn input_size(&self) -> usize { self.layers.first().expect("can not get output size of empty network").nodes_in() }
    fn output_size(&self) -> usize { self.layers.last().expect("can not get output size of empty network").nodes_out() }
    fn num_layers(&self) -> usize { self.layers.len() }

    fn calculate(&self, input: DVector<T>) -> DVector<T> {
        self.layers.iter().fold(input, |acc, v| {
            v.calculate(&acc, &self.activation_fn)
        })
    }

    fn calculate_layer_outputs(&self, input: DVector<T>) -> Vec<DVector<T>> {
        let mut res = Vec::with_capacity(self.num_layers());
        res.push(input);

        for layer in self.layers.iter() { res.push(layer.calculate(res.last().unwrap(), &self.activation_fn)) }

        res
    }

    fn classify(&self, input: Vec<T>) -> usize {
        let input = DVector::from_vec(input);
        self.calculate(input).into_iter()
            .enumerate()
        .reduce(|acc, v| if acc.1 > v.1 { acc } else { v })
        .expect("cannot classify with zero-sized last layer").0
    }

    fn train_data_point(&mut self, learn_rate: T, data_point: DataPoint<T>) {
        let outputs = self.calculate_layer_outputs(DVector::from_vec(data_point.value));

        // let final_output = self.calculate(DVector::from_vec(data_point.value));
        let final_output = outputs.last().expect("cannot train network with 0 layers");
        // let mut errors = Vec::with_capacity(self.num_layers());

        let out_err = DVector::from_iterator(final_output.len(), final_output.iter().zip(data_point.expected.iter()).map(|(output, expected)| {
            self.cost_fn.calc_derivative(*output, *expected)
        }));
        // errors.push(out_err);

        // for layer in self.layers.iter().skip(1).rev() {
        //     let err = layer.weights.tr_mul(errors.last().unwrap());
        //     errors.push(err);
        // }

        self.layers[1..].iter_mut().rev()
            .zip(outputs.windows(2).rev())
        .fold(out_err, |err, (v, outputs)| {
            // println!("{}, {}", outputs[0].nrows(), outputs[1].nrows());
            //           5-v               10-v
            let change = err.component_mul(&outputs[1].map(|v| v * (T::one() - v))) * outputs[0].transpose();
            v.weights += change * learn_rate;

            v.weights.tr_mul(&err)
        });
    }
}

impl<T, A, C> Network<T, A, C>
where
    T: NetworkValue,
    A: ActivationFn<T> + Default,
    C: CostFn<T> + Default,
{
    fn new(layers: &[usize]) -> Self {
        Self::new_fn(layers, Default::default(), Default::default())
    }
}

impl<T, A, C> Network<T, A, C>
where
    T: NetworkValue,
    StandardNormal: Distribution<T>,
    A: ActivationFn<T>,
    C: CostFn<T>,
{
    fn new_random_fn(layers: &[usize], activation_fn: A, cost_fn: C) -> Self {
        let layers = layers.windows(2)
            .map(|v| Layer::new_random(v[0], v[1]))
        .collect();
        Self { layers, activation_fn, cost_fn }
    }
}

impl<T, A, C> Network<T, A, C>
where
    T: NetworkValue,
    StandardNormal: Distribution<T>,
    A: ActivationFn<T> + Default,
    C: CostFn<T> + Default,
{
    fn new_random(layers: &[usize]) -> Self {
        Self::new_random_fn(layers, Default::default(), Default::default())
    }
}

pub struct DataPoint<T> {
    value: Vec<T>,
    expected: Vec<T>,
}

impl<T> DataPoint<T> {
    fn new(value: Vec<T>, expected: Vec<T>) -> Self {
        Self { value, expected }
    }
}

enum MnistData {
    U8(Vec<u8>),
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    F32(Vec<f32>),
    F64(Vec<f32>),
}

enum Dimension {
    D1(usize),
    D2(usize, usize),
    D3(usize, usize, usize),
    Dn(Vec<usize>),
}


fn main() {
    let labels: Vec<u8> = read_idx_path("./train-data/train-labels.idx1-ubyte");
    let images: Vec<Vec<u8>> = read_idx_path("./train-data/train-images.idx3-ubyte");
    dbg!(labels.len(), images[0].len());

    let mut net: Network<f32, Sigmoid, SquaredError> = Network::new_random(&[images[0].len(), 10, 10, 10]);
    let learn_rate = 0.1;

    for (img, label) in images.iter().zip(labels.iter()) {
        let mut expected = vec![0.01; 10];
        expected[*label as usize] = 0.99;

        let p = DataPoint::new(img.iter().map(|v| (*v as f32) / 255.0 * 0.98 + 0.01).collect(), expected);
        net.train_data_point(learn_rate, p);
    }

    println!("finished training");

    let test_labels: Vec<u8> = read_idx_path("./train-data/t10k-labels.idx1-ubyte");
    let test_images: Vec<Vec<u8>> = read_idx_path("./train-data/t10k-images.idx3-ubyte");

    let acc = (test_images.iter().zip(test_labels.iter()).filter(|(img, label)| {
        let res = net.classify(img.iter().map(|v| (*v as f32) / 255.0 * 0.98 + 0.01).collect());
        (res == **label as _)
    }).count() as f32) / (test_images.len() as f32);
    println!("acc: {acc}");
}

fn net_main() {
    let mut net: Network<f32, Sigmoid, SquaredError> = Network::new_random(&[5, 10, 10, 5]);

    // let res = net.classify(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    // println!("res: {res}");

    let mut rng = rand::thread_rng();
    for _ in (0..10000) {
        let mut p = DataPoint::new(vec![0.1; net.input_size()], vec![0.1; net.output_size()]);
        let v = rng.gen_range(0..net.input_size());
        p.expected[v] = 0.99;
        p.value[v] = 0.99;

        net.train_data_point(0.1, p);
    }

    let res = net.calculate(dvector![0.9, 0.1, 0.1, 0.1, 0.1]);
    println!("res: {res}");

    println!("{}", net.layers[0].weights);
}



