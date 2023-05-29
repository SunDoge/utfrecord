use std::io::BufReader;

use dlpark::tensor::ManagerCtx;
use fastdata_tfrecord::sync_reader::TfrecordReader;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::example::parse_tfrecord;

#[pyclass]
pub struct KanalReceiver {
    inner: kanal::Receiver<Vec<u8>>,
    handle: std::thread::JoinHandle<()>,
}

#[pymethods]
impl KanalReceiver {
    #[new]
    fn new(path: &str, channel_size: usize, check_integrity: bool) -> Self {
        let (sender, receiver) = kanal::bounded(channel_size);
        let file = std::fs::File::open(path).unwrap();
        let buf_reader = BufReader::new(file);
        let reader = TfrecordReader::new(buf_reader, check_integrity);
        let handle = std::thread::spawn(move || {
            for record in reader {
                sender.send(record.unwrap()).unwrap();
            }
        });
        Self {
            inner: receiver,
            handle,
        }
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<ManagerCtx<Vec<u8>>> {
        slf.inner.next().map(|x| x.into())
    }
}

#[pyclass]
pub struct KanalReceiverParsed {
    inner: kanal::Receiver<Vec<u8>>,
    keys: Vec<String>,
    handle: std::thread::JoinHandle<()>,
}

#[pymethods]
impl KanalReceiverParsed {
    #[new]
    fn new(
        paths: Vec<String>,
        cycle: bool,
        channel_size: usize,
        keys: Vec<String>,
        check_integrity: bool,
    ) -> Self {
        let (sender, receiver) = kanal::bounded(channel_size);

        let handle = std::thread::spawn(move || {
            if cycle {
                for path in paths.iter().cycle() {
                    let file = std::fs::File::open(path).unwrap();
                    let buf_reader = BufReader::new(file);
                    let reader = TfrecordReader::new(buf_reader, check_integrity);
                    for record in reader {
                        sender.send(record.unwrap()).unwrap();
                    }
                }
            } else {
                for path in paths.iter() {
                    let file = std::fs::File::open(path).unwrap();
                    let buf_reader = BufReader::new(file);
                    let reader = TfrecordReader::new(buf_reader, check_integrity);
                    for record in reader {
                        sender.send(record.unwrap()).unwrap();
                    }
                }
            }
        });
        Self {
            inner: receiver,
            handle,
            keys,
        }
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'a>(mut slf: PyRefMut<'a, Self>) -> Option<Py<PyDict>> {
        slf.inner
            .next()
            .map(|buf| Python::with_gil(|py| parse_tfrecord(py, &buf, &slf.keys).into_py(py)))
    }
}
