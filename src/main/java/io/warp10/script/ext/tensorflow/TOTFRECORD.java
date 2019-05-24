//
//   Copyright 2019  SenX S.A.S.
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
//

package io.warp10.script.ext.tensorflow;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import org.apache.hadoop.util.PureJavaCrc32C;
import org.tensorflow.example.Example;
import org.tensorflow.example.SequenceExample;

import io.warp10.script.NamedWarpScriptFunction;
import io.warp10.script.WarpScriptException;
import io.warp10.script.WarpScriptStack;
import io.warp10.script.WarpScriptStackFunction;

public class TOTFRECORD extends NamedWarpScriptFunction implements WarpScriptStackFunction {
  
  static final int MASK_DELTA = 0xa282ead8;
  
  public TOTFRECORD(String name) {
    super(name);
  }

  @Override
  public Object apply(WarpScriptStack stack) throws WarpScriptException {
    
    Object top = stack.pop();
    
    if (!(top instanceof Example) && !(top instanceof SequenceExample)) {
      throw new WarpScriptException(getName() + " expects a TensorFlow Example or SequenceExample on top of the stack.");
    }

    byte[] bytes;
    
    if (top instanceof Example) {
      bytes = ((Example) top).toByteArray();
    } else {
      bytes = ((SequenceExample) top).toByteArray();
    }

    stack.push(toTFRecord(bytes));

    return stack;
  }
  
  public static final byte[] toTFRecord(byte[] bytes) {
    //
    // TFRecord format:
    // uint64 length
    // uint32 masked_crc32_of_length
    // byte   data[length]
    // uint32 masked_crc32_of_data
    //
    
    byte[] tfrecord = new byte[bytes.length + 8 + 8];    
    ByteBuffer tfrecordbb = ByteBuffer.wrap(tfrecord);
    tfrecordbb.order(ByteOrder.LITTLE_ENDIAN);
    
    byte[] len = new byte[8];
    ByteBuffer bb = ByteBuffer.wrap(len);
    bb.order(ByteOrder.LITTLE_ENDIAN);
    bb.putLong(bytes.length);
    
    PureJavaCrc32C crc32c = new PureJavaCrc32C();    
    crc32c.update(len, 0, 8);
    int crc = ((int) crc32c.getValue());
    //  Rotate right by 15 bits and add a constant.
    crc = ((crc >>> 15) | (crc << 17)) + MASK_DELTA;
    
    tfrecordbb.put(len);
    tfrecordbb.putInt(crc);

    crc32c = new PureJavaCrc32C();    
    crc32c.update(bytes, 0, bytes.length);
    crc = ((int) crc32c.getValue());
    //  Rotate right by 15 bits and add a constant.
    crc = ((crc >>> 15) | (crc << 17)) + MASK_DELTA;
    
    tfrecordbb.put(bytes);
    tfrecordbb.putInt(crc);

    return tfrecord;
  }
}
