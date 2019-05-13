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

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.tensorflow.example.BytesList;
import org.tensorflow.example.Feature;
import org.tensorflow.example.FeatureList;
import org.tensorflow.example.FeatureLists;
import org.tensorflow.example.Features;
import org.tensorflow.example.FloatList;
import org.tensorflow.example.Int64List;
import org.tensorflow.example.SequenceExample;

import com.google.common.base.Charsets;
import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;

import io.warp10.script.NamedWarpScriptFunction;
import io.warp10.script.WarpScriptException;
import io.warp10.script.WarpScriptStack;
import io.warp10.script.WarpScriptStackFunction;

public class TOTFSEQEXAMPLE extends NamedWarpScriptFunction implements WarpScriptStackFunction {

  public static final String CONTEXT_KEY = "context";
  public static final String FEATURELISTS_KEY = "featurelists";
  
  public TOTFSEQEXAMPLE(String name) {
    super(name);
  }

  @Override
  public Object apply(WarpScriptStack stack) throws WarpScriptException {
    
    Object top = stack.pop();
    
    if (!(top instanceof Map) && !(top instanceof byte[])) {
      throw new WarpScriptException(getName() + " expects a map or a serialized TensorFlow SequenceExample on top of the stack.");
    }
    
    SequenceExample seqexample = null;

    if (top instanceof Map) {
      Map<Object,Object> map = (Map<Object,Object>) top;
      
      Features.Builder context = Features.newBuilder();

      Map<Object,Object> contextmap = (Map<Object,Object>) map.getOrDefault(CONTEXT_KEY, new HashMap<Object,Object>());
      
      for (Entry<Object,Object> entry: contextmap.entrySet()) {
        if (!(entry.getKey() instanceof String)) {
          throw new WarpScriptException(getName() + " expects context keys to be strings.");
        }
        
        Object value = entry.getValue();
        String key = entry.getKey().toString();

        context.putFeature(key, encodeFeature(getName(), value));
      }
      
      Map<Object,Object> featureslists = (Map<Object,Object>) map.getOrDefault(FEATURELISTS_KEY, new HashMap<Object,Object>());
      
      FeatureLists.Builder flbuilder = FeatureLists.newBuilder();

      for (Entry<Object,Object> entry: featureslists.entrySet()) {
        if (!(entry.getKey() instanceof String)) {
          throw new WarpScriptException(getName() + " expects featurelists keys to be strings.");
        }
        
        String key = entry.getKey().toString();
        Object value = entry.getValue();
        
        FeatureList.Builder fl = FeatureList.newBuilder();
        
        if (value instanceof List) {
          // Check if one of the elements is a list, if not consider it is a simple feature
          boolean containsList = false;
          for (Object val: (List<Object>) value) {
            if (val instanceof List) {
              containsList = true;
              break;
            }
          }
          if (containsList) {
            for (Object val: (List<Object>) value) {
              fl.addFeature(encodeFeature(getName(), val));
            }
          } else {
            fl.addFeature(encodeFeature(getName(), value));
          }
        } else {
          fl.addFeature(encodeFeature(getName(), value));
        }
        
        flbuilder.putFeatureList(key, fl.build());
      }
      
      seqexample = SequenceExample.newBuilder().setContext(context).setFeatureLists(flbuilder).build();      
    } else {
      try {
        seqexample = SequenceExample.parseFrom((byte[]) top);
      } catch (InvalidProtocolBufferException ipbe) {
        throw new WarpScriptException(getName() + " encoutered an error while parsing TensorFlow SequenceExample.");
      }
    }

    stack.push(seqexample);
    
    return stack;
  }

  public static Feature encodeFeature(String name, Object value) throws WarpScriptException {
    if (value instanceof Long || value instanceof Integer || value instanceof Byte || value instanceof BigInteger) {
      Int64List il = Int64List.newBuilder().addValue(((Number) value).longValue()).build();
      return Feature.newBuilder().setInt64List(il).build();
    } else if (value instanceof Double || value instanceof Float || value instanceof BigDecimal) {
      FloatList fl = FloatList.newBuilder().addValue(((Number) value).floatValue()).build();        
      return Feature.newBuilder().setFloatList(fl).build();
    } else if (value instanceof byte[]) {
      BytesList bl = BytesList.newBuilder().addValue(ByteString.copyFrom((byte[]) value)).build();
      return Feature.newBuilder().setBytesList(bl).build();
    } else if (value instanceof String) {
      BytesList bl = BytesList.newBuilder().addValue(ByteString.copyFrom((String) value, Charsets.UTF_8)).build();        
      return Feature.newBuilder().setBytesList(bl).build();
    } else if (value instanceof List) {
      List<Object> l = (List<Object>) value;
      
      boolean typed = false;
      
      Int64List.Builder ib = null;
      FloatList.Builder fb = null;
      BytesList.Builder bb = null;
      
      for (Object elt: l) {
        if (elt instanceof Long || elt instanceof Integer || elt instanceof Byte || elt instanceof BigInteger) {
          if (typed && null == ib) {
            throw new WarpScriptException(name + " expects value lists to contains elements of the same type.");
          }
          if (null == ib) {
            typed = true;
            ib = Int64List.newBuilder();
          }
          ib.addValue(((Number) elt).longValue());
        } else if (elt instanceof Double || elt instanceof Float || elt instanceof BigDecimal) {
          if (typed && null == fb) {
            throw new WarpScriptException(name + " expects value lists to contains elements of the same type.");
          }
          if (null == fb) {
            typed = true;
            fb = FloatList.newBuilder();
          }
          fb.addValue(((Number) elt).floatValue());
        } else if (elt instanceof byte[]) {
          if (typed && null == bb) {
            throw new WarpScriptException(name + " expects value lists to contains elements of the same type.");
          }
          if (null == bb) {
            typed = true;
            bb = BytesList.newBuilder();
          }
          bb.addValue(ByteString.copyFrom((byte[]) elt));
        } else if (elt instanceof String) {
          if (typed && null == bb) {
            throw new WarpScriptException(name + " expects value lists to contains elements of the same type.");
          }
          if (null == bb) {
            typed = true;
            bb = BytesList.newBuilder();
          }
          bb.addValue(ByteString.copyFrom((String) elt, Charsets.UTF_8));
        }
      }
      
      if (null != bb) {
        return Feature.newBuilder().setBytesList(bb).build();
      } else if (null != ib) {
        return Feature.newBuilder().setInt64List(ib).build();
      } else if (null != fb) {
        return Feature.newBuilder().setFloatList(fb).build();          
      } else {
        throw new WarpScriptException(name + " encountered an empty value list.");
      }                
    } else {
      throw new WarpScriptException(name + " invalid feature value type.");
    }
  }
}
