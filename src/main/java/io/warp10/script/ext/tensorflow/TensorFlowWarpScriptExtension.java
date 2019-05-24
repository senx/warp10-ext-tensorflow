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

import java.util.HashMap;
import java.util.Map;

import io.warp10.warp.sdk.WarpScriptExtension;

public class TensorFlowWarpScriptExtension extends WarpScriptExtension {
  
  private static Map<String,Object> functions;
  
  static {
    functions = new HashMap<String,Object>();   
    
    functions.put("->TFEXAMPLE",  new TOTFEXAMPLE("->TFEXAMPLE"));
    functions.put("->TFRECORD",  new TOTFRECORD("->TFRECORD"));
    functions.put("TFRECORD->",  new TFRECORDTO("TFRECORD->"));
    functions.put("TFEXAMPLE->",  new TFEXAMPLETO("TFEXAMPLE->"));
    functions.put("->TFSEQEXAMPLE", new TOTFSEQEXAMPLE("->TFSEQEXAMPLE"));
    functions.put("TFSEQEXAMPLE->", new TFSEQEXAMPLETO("TFSEQEXAMPLE->"));
  }
  
  @Override
  public Map<String, Object> getFunctions() {
    return functions;
  }  
}
