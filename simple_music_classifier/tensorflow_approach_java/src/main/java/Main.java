import org.tensorflow.*;

import java.util.Random;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.List;
import java.lang.Integer;

// mvn compile exec:java

public class Main {
    public static void main(String[] args) throws Exception {
      // some "globals"
      Random rand = new Random();
      final int kNumReps = 20000;
      final int kChunkSize = 6300;
      final int kBatchSize = 1;
      final long[] shape = new long[] {kBatchSize, kChunkSize};
      final String model_path = "../tensorflow_approach_python/saved_models/29_Nov_2017_02h47m28s";
      // load bundle (graph, session...)
      try (SavedModelBundle model = SavedModelBundle.load(model_path, "my_model")){
          Graph g = model.graph();
          Session sess = model.session();
          Output<Float> data_ph = g.operation("data_placeholder").output(0);
          Output<Float> preds = g.operation("preds").output(0);
          // LOOP
          for (int i=0; i<kNumReps; i++){
            if(i%1000==0){
              System.out.println("Model evaluated "+ i + " times");
            }
            FloatBuffer buf = FloatBuffer.allocate(kChunkSize);
            for (int j=0; j<kChunkSize; ++j) {
              buf.put(rand.nextFloat()*2-1);
            }
            buf.flip();
            try (Tensor<Float> signal = Tensor.create(shape, buf)){
                // make predictions
                List<Tensor<?>> predictions = sess.runner().feed(data_ph, signal).fetch(preds).run();
                // // print predictions
                // for(Tensor t : predictions) {
                //   IntBuffer result = IntBuffer.allocate(t.numElements());
                //   t.writeTo(result);
                //   for (Integer k : result.array()){
                //     System.out.println(k);
                //   }
                // }
              }
          }
        }
    }
}
