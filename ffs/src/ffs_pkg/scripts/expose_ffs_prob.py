#!/usr/bin/env python3
"""Expose FFS's disparity softmax as a second post-runner ONNX output."""

import argparse
from copy import deepcopy

import onnx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_onnx")
    parser.add_argument("output_onnx")
    args = parser.parse_args()

    model = onnx.shape_inference.infer_shapes(onnx.load(args.input_onnx))
    candidates = [
        node for node in model.graph.node
        if node.op_type == "Softmax"
        and any(attr.name == "axis" and attr.i == 1 for attr in node.attribute)
        and any(input_name.startswith("/Squeeze") for input_name in node.input)
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one disparity softmax, found {len(candidates)}")
    source = candidates[0].output[0]
    info = deepcopy(next(info for info in model.graph.value_info if info.name == source))
    info.name = "prob"
    model.graph.node.append(onnx.helper.make_node("Identity", [source], ["prob"]))
    model.graph.output.append(info)
    onnx.checker.check_model(model)
    onnx.save(model, args.output_onnx)


if __name__ == "__main__":
    main()
