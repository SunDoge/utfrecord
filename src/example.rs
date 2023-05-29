use std::collections::HashMap;

use dlpark::prelude::*;
use fastdata_tfrecord::tensorflow::Example;
use pyo3::{types::PyDict, Py, Python};

pub fn parse_tfrecord<'a>(py: Python<'a>, example_bytes: &[u8], keys: &[&str]) -> &PyDict {
    // let example
    unimplemented!()
}
