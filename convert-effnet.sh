#!/bin/sh
# Rebuilds models/discogs-effnet.onnx from Essentia's TensorFlow graph (see NOTICE):
# tf2onnx, then every Reshape target whose first entry is the graph's fixed batch
# of 64 gets -1, so a track's patches go through in one batch of any size. Checks
# the result against TensorFlow before writing it. Needs Docker.
set -eu
cd "$(dirname "$0")"
docker run --rm -v "$PWD":/w -w /w python:3.11-slim sh -c '
pip install -q --no-cache-dir tensorflow==2.21.0 tf2onnx==1.17.0 onnx==1.23.2 onnxruntime==1.31.1 "numpy<2" >/dev/null
python3 - <<EOF
import hashlib, subprocess, sys, tempfile, urllib.request
import numpy as np, onnx, onnxruntime as ort, tensorflow as tf
from onnx import numpy_helper

URL = "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb"
SHA = "3ed9af50d5367c0b9c795b294b00e7599e4943244f4cbd376869f3bfc87721b1"
pb = urllib.request.urlopen(URL, timeout=300).read()
assert hashlib.sha256(pb).hexdigest() == SHA, "unexpected source graph"
tmp = tempfile.mkdtemp()
open(f"{tmp}/in.pb", "wb").write(pb)
subprocess.run([sys.executable, "-m", "tf2onnx.convert", "--graphdef", f"{tmp}/in.pb",
                "--inputs", "serving_default_melspectrogram:0[-1,128,96]",
                "--outputs", "PartitionedCall:0,PartitionedCall:1", "--opset", "17",
                "--output", f"{tmp}/out.onnx"], check=True, capture_output=True)

m = onnx.load(f"{tmp}/out.onnx")
inits = {i.name: i for i in m.graph.initializer}
consts = {n.output[0]: n.attribute[0] for n in m.graph.node if n.op_type == "Constant"}
for n in m.graph.node:
    if n.op_type != "Reshape":
        continue
    if n.input[1] in inits:
        t = inits[n.input[1]]
        v = numpy_helper.to_array(t).copy()
        if v.size and v[0] == 64:
            v[0] = -1
            t.CopyFrom(numpy_helper.from_array(v, t.name))
    elif n.input[1] in consts:
        a = consts[n.input[1]]
        v = numpy_helper.to_array(a.t).copy()
        if v.size and v[0] == 64:
            v[0] = -1
            a.t.CopyFrom(numpy_helper.from_array(v))
for o in m.graph.output:
    o.type.tensor_type.shape.dim[0].dim_param = "batch"
del m.graph.value_info[:]
onnx.save(m, f"{tmp}/patched.onnx")

x = np.random.default_rng(0).uniform(0, 4, (64, 128, 96)).astype(np.float32)
gd = tf.compat.v1.GraphDef()
gd.ParseFromString(pb)
g = tf.Graph()
with g.as_default():
    tf.import_graph_def(gd, name="")
with tf.compat.v1.Session(graph=g) as s:
    want = s.run(["PartitionedCall:0", "PartitionedCall:1"], {"serving_default_melspectrogram:0": x})
sess = ort.InferenceSession(f"{tmp}/patched.onnx")
for b in (1, 20, 64):
    got = sess.run(None, {"serving_default_melspectrogram:0": x[:b]})
    err = max(np.abs(g - w[:b]).max() for g, w in zip(got, want))
    assert err < 1e-4, (b, err)
open("models/discogs-effnet.onnx", "wb").write(open(f"{tmp}/patched.onnx", "rb").read())
print("models/discogs-effnet.onnx", hashlib.sha256(open("models/discogs-effnet.onnx", "rb").read()).hexdigest())
EOF
'
