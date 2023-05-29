use std::collections::HashMap;

use dlpark::prelude::*;
use fastdata_tfrecord::tensorflow::{
    feature::Kind, BytesList, Example, Feature, FloatList, Int64List,
};
use pyo3::{types::PyDict, IntoPy, PyObject, Python};

pub fn parse_tfrecord<'a>(py: Python<'a>, example_bytes: &[u8], keys: &[String]) -> &'a PyDict {
    let mut example = Example::from_bytes(example_bytes).unwrap();
    let dic = PyDict::new(py);
    for key in keys {
        let feat = example
            .features
            .as_mut()
            .unwrap()
            .feature
            .remove(key)
            .unwrap();

        let value = feature_to_pyobject(py, feat);
        dic.set_item(key, value).unwrap();
    }
    dic
}

fn feature_to_pyobject<'a>(py: Python<'a>, feat: Feature) -> PyObject {
    match feat.kind {
        Some(Kind::BytesList(BytesList { value })) => {
            let v: Vec<_> = value
                .into_iter()
                .map(|x| ManagerCtx::from(x).into_py(py))
                .collect();
            v.into_py(py)
        }
        Some(Kind::FloatList(FloatList { value })) => ManagerCtx::from(value).into_py(py),
        Some(Kind::Int64List(Int64List { value })) => ManagerCtx::from(value).into_py(py),
        None => panic!("none"),
    }
}
