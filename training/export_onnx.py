#!/usr/bin/env python3
"""Export the transparent logistic artifact to a minimal ONNX graph.

The exporter fails with an actionable message when the optional `onnx` package
is unavailable; it never writes a fake model file.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path

from qxvision.model import LogisticModel, FEATURE_SCHEMA_VERSION

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument("--registry",required=True); ap.add_argument("--output",required=True); args=ap.parse_args(argv)
    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError as e:
        raise SystemExit("ONNX export requires the free optional package: python -m pip install onnx") from e
    registry=Path(args.registry); model=LogisticModel.from_json(json.loads(registry.joinpath("logistic.json").read_text()))
    import numpy as np
    x=helper.make_tensor_value_info("features",TensorProto.FLOAT,[None,len(model.weights)])
    y=helper.make_tensor_value_info("probability_up",TensorProto.FLOAT,[None,1])
    w=numpy_helper.from_array(np.asarray(model.weights,dtype=np.float32).reshape(len(model.weights),1),"weights")
    b=numpy_helper.from_array(np.asarray([model.bias],dtype=np.float32),"bias")
    graph=helper.make_graph([helper.make_node("MatMul",["features","weights"],["linear"]),helper.make_node("Add",["linear","bias"],["logit"]),helper.make_node("Sigmoid",["logit"],["probability_up"])],"qxvision_logistic",[x],[y],[w,b])
    model_proto=helper.make_model(graph,producer_name="qxvision",opset_imports=[helper.make_opsetid("",13)]); onnx.checker.check_model(model_proto)
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); onnx.save(model_proto,out)
    print(f"wrote {out} ({out.stat().st_size} bytes); schema={FEATURE_SCHEMA_VERSION}")
    try:
        import onnxruntime as ort
        sess=ort.InferenceSession(str(out),providers=["CPUExecutionProvider"]); sample=np.zeros((1,len(model.weights)),dtype=np.float32); value=float(sess.run(None,{"features":sample})[0][0][0]); expected=1/(1+math.exp(-model.bias)); assert abs(value-expected)<1e-5, (value,expected); print("onnx parity: PASS")
    except ImportError: print("onnxruntime not installed; graph checked, runtime parity deferred")
    return 0
if __name__=="__main__": main()
