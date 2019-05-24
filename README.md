## WarpScript™ TensorFlow extension

This extension adds functions to the WarpScript™ language to manipulate [TensorFlow](https://tensorflow.org/) `TFRecords`.

`TFRecord` is the recommended format for storing and feeding data into models for both training and inference. It is a compact binary format which can be used in TensorFlow without needing to convert back and forth to Python Numpy arrays.

The use of `TFRecords` therefore brings imporved performance and efficiency to your Machine Learning pipeline.

## Installation

Once you have downloaded the `.jar` file of the extension via the [WarpFleet](https://warpfleet.senx.io) `wf` tool you need to enable the extension by adding the following line to your Warp 10™ configuration:

```
  warpscript.extension.tensorflow = io.warp10.script.ext.tensorflow.TensorFlowWarpScriptExtension
```

You can then restart your Warp 10™ instance.

## New functions

The extension adds functions to create [`tf.train.Example`](https://www.tensorflow.org/tutorials/load_data/tf_records#tfexample) and [`tf.train.SequenceExample`](https://www.tensorflow.org/api_docs/python/tf/train/SequenceExample) instances and pack them in [`TFRecord`](https://www.tensorflow.org/alpha/tutorials/load_data/tf_records#tfrecords_format_details).

It also adds functions to extract information from those formats.