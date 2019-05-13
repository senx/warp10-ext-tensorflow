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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.tensorflow.example.Feature;
import org.tensorflow.example.FeatureList;
import org.tensorflow.example.FeatureLists;
import org.tensorflow.example.Features;
import org.tensorflow.example.SequenceExample;

import io.warp10.script.NamedWarpScriptFunction;
import io.warp10.script.WarpScriptException;
import io.warp10.script.WarpScriptStack;
import io.warp10.script.WarpScriptStackFunction;

public class TFSEQEXAMPLETO extends NamedWarpScriptFunction implements WarpScriptStackFunction {
  public TFSEQEXAMPLETO(String name) {
    super(name);
  }
  
  @Override
  public Object apply(WarpScriptStack stack) throws WarpScriptException {
    
    Object top = stack.pop();
    
    if (!(top instanceof SequenceExample)) {
      throw new WarpScriptException(getName() + " expects a TensorFlow SequenceExample on top of the stack.");
    }

    SequenceExample seqex = (SequenceExample) top;
    
    Map<String,Object> map = new HashMap<String,Object>();
    
    Features context = seqex.getContext();
    Map<String,Feature> features = context.getFeatureMap();
    
    Map<String,Object> fmap = new HashMap<String,Object>(features.size());
    
    for (Entry<String,Feature> entry: features.entrySet()) {
      Feature feature = entry.getValue();
      fmap.put(entry.getKey(), TFEXAMPLETO.decodeFeature(feature));      
    }

    map.put(TOTFSEQEXAMPLE.CONTEXT_KEY, fmap);
    
    FeatureLists flists = seqex.getFeatureLists();
    
    Map<String,FeatureList> flmap = flists.getFeatureListMap();
    
    Map<String,Object> featureLists = new HashMap<String,Object>();
    
    for (Entry<String,FeatureList> entry: flmap.entrySet()) {
      List<Feature> flist = entry.getValue().getFeatureList();
      
      ArrayList<Object> feats = new ArrayList<Object>(flist.size());
      
      for (Feature feat: flist) {
        feats.add(TFEXAMPLETO.decodeFeature(feat));
      }
      
      featureLists.put(entry.getKey(), feats);
    }
    
    map.put(TOTFSEQEXAMPLE.FEATURELISTS_KEY, featureLists);
    
    stack.push(map);
    
    return stack;
  }
}
