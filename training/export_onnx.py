#!/usr/bin/env python3
"""Export a candidate logistic artifact to ONNX and verify numerical parity."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path

from qxvision.model import LogisticModel
from qxvision.features import FEATURE_SCHEMA_VERSION


def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument("--registry",required=True); ap.add_argument("--output",required=True); args=ap.parse_args(argv)
    registry=Path(args.registry); out=Path(args.output); parity_path=out.parent/"onnx_parity.json"
    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError as exc:
        parity_path.write_text(json.dumps({"passed":False,"reason":"ONNX_PACKAGE_UNAVAILABLE"},indent=2)); raise SystemExit("ONNX export requires the free optional package: python -m pip install onnx") from exc
    try:
        import numpy as np
    except ImportError as exc:
        parity_path.write_text(json.dumps({"passed":False,"reason":"NUMPY_PACKAGE_UNAVAILABLE"},indent=2)); raise SystemExit("ONNX export requires numpy") from exc
    model=LogisticModel.from_json(json.loads(registry.joinpath("logistic.json").read_text()))
    metadata=json.loads(registry.joinpath("metadata.json").read_text())
    if metadata.get("feature_schema_version")!=FEATURE_SCHEMA_VERSION: raise SystemExit("FEATURE_SCHEMA_MISMATCH")
    x=helper.make_tensor_value_info("features",TensorProto.FLOAT,[None,len(model.weights)]); y=helper.make_tensor_value_info("probability_up",TensorProto.FLOAT,[None,1])
    w=numpy_helper.from_array(np.asarray(model.weights,dtype=np.float32).reshape(len(model.weights),1),"weights"); b=numpy_helper.from_array(np.asarray([model.bias],dtype=np.float32),"bias")
    graph=helper.make_graph([helper.make_node("MatMul",["features","weights"],["linear"]),helper.make_node("Add",["linear","bias"],["logit"]),helper.make_node("Sigmoid",["logit"],["probability_up"])],"qxvision_logistic",[x],[y],[w,b])
    model_proto=helper.make_model(graph,producer_name="qxvision",opset_imports=[helper.make_opsetid("",13)]); onnx.checker.check_model(model_proto); out.parent.mkdir(parents=True,exist_ok=True); onnx.save(model_proto,out)
    status={"passed":False,"tolerance":1e-5,"sample_count":0,"onnx_path":str(out)}
    try:
        import onnxruntime as ort
        sess=ort.InferenceSession(str(out),providers=["CPUExecutionProvider"]); samples=np.zeros((min(5,len(model.weights)),len(model.weights)),dtype=np.float32); samples[0,:]=np.linspace(-.5,.5,len(model.weights),dtype=np.float32); source=[model.probability(row.tolist()) for row in samples]; runtime=[float(value[0]) for value in sess.run(["probability_up"],{"features":samples})[0]]; error=max(abs(a-b) for a,b in zip(source,runtime)); status.update({"passed":error<=status["tolerance"],"sample_count":len(source),"max_abs_error":error,"source_probabilities":source,"onnx_probabilities":runtime})
    except ImportError: status["reason"]="ONNXRUNTIME_PACKAGE_UNAVAILABLE"
    metadata["onnx_sha256"] = __import__("hashlib").sha256(out.read_bytes()).hexdigest()
    metadata["onnx_size_bytes"] = out.stat().st_size
    metadata["onnx_parity_file"] = str(parity_path.name)
    registry.joinpath("metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    parity_path.write_text(json.dumps(status,indent=2),encoding="utf-8")
    if not status["passed"]: raise SystemExit("ONNX_PARITY_NOT_VERIFIED: " + status.get("reason",f"max_abs_error={status.get('max_abs_error')}"))
    print(f"onnx parity: PASS; max_abs_error={status['max_abs_error']}"); return 0

if __name__=="__main__": raise SystemExit(main())
