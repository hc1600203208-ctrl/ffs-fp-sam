#!/bin/bash

/usr/src/tensorrt/bin/trtexec --onnx=/home/hc/weizi/ffs+fp+sam/fp/models/scorer_hwc.onnx \
                              --minShapes=render_input:1x160x160x6,transf_input:1x160x160x6 \
                              --optShapes=render_input:252x160x160x6,transf_input:252x160x160x6 \
                              --maxShapes=render_input:252x160x160x6,transf_input:252x160x160x6 \
                              --saveEngine=/home/hc/weizi/ffs+fp+sam/fp/models/scorer_hwc_dynamic_fp32_5060.engine

/usr/src/tensorrt/bin/trtexec --onnx=/home/hc/weizi/ffs+fp+sam/fp/models/refiner_hwc.onnx \
                              --minShapes=render_input:1x160x160x6,transf_input:1x160x160x6 \
                              --optShapes=render_input:252x160x160x6,transf_input:252x160x160x6 \
                              --maxShapes=render_input:252x160x160x6,transf_input:252x160x160x6 \
                              --saveEngine=/home/hc/weizi/ffs+fp+sam/fp/models/refiner_hwc_dynamic_fp32_5060.engine
