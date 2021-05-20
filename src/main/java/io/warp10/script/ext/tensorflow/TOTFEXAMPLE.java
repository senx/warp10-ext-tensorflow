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

import java.util.Map;
import java.util.Map.Entry;

import org.tensorflow.example.Example;
import org.tensorflow.example.Features;

import com.google.protobuf.InvalidProtocolBufferException;

import io.warp10.script.NamedWarpScriptFunction;
import io.warp10.script.WarpScriptException;
import io.warp10.script.WarpScriptStack;
import io.warp10.script.WarpScriptStackFunction;

public class TOTFEXAMPLE extends NamedWarpScriptFunction implements WarpScriptStackFunction {
  
  public TOTFEXAMPLE(String name) {
    super(name);
  }

  @Override
  public Object apply(WarpScriptStack stack) throws WarpScriptException {
    
    Object top = stack.pop();
    
    if (!(top instanceof Map) && !(top instanceof byte[])) {
      throw new WarpScriptException(getName() + " expects a map or a serialized TensorFlow Example on top of the stack.");
    }
    
    Example example = null;

    if (top instanceof Map) {
      Map<Object,Object> map = (Map<Object,Object>) top;
      
      Features.Builder features = Features.newBuilder();
      
      for (Entry<Object,Object> entry: map.entrySet()) {
        if (!(entry.getKey() instanceof String)) {
          throw new WarpScriptException(getName() + " expects map keys to be strings.");
        }
        
        Object value = entry.getValue();
        String key = entry.getKey().toString();

        features.putFeature(key, TOTFSEQEXAMPLE.encodeFeature(getName(), value));
      }
      
      example = Example.newBuilder().setFeatures(features).build();      
    } else {
      try {
        example = Example.parseFrom((byte[]) top);
      } catch (InvalidProtocolBufferException ipbe) {
        throw new WarpScriptException(getName() + " encoutered an error while parsing TensorFlow Example.");
      }
    }

    stack.push(example);
    
    return stack;
  }
}
