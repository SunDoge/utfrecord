pub mod example;
pub mod reader;

use pyo3::prelude::*;
use reader::{IoUringTfrecordReader, KanalReceiver, KanalReceiverParsed};

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn utfrecord(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_class::<KanalReceiver>()?;
    m.add_class::<KanalReceiverParsed>()?;
    m.add_class::<IoUringTfrecordReader>()?;
    Ok(())
}
