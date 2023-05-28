use std::io::BufReader;

use dlpark::tensor::ManagerCtx;
use fastdata_tfrecord::sync_reader::TfrecordReader;
use pyo3::prelude::*;

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
