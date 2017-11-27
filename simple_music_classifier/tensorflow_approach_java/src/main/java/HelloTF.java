import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;

// TF java examples: https://github.com/loretoparisi/tensorflow-java

public class HelloTF {
  public static void main(String[] args) throws Exception {
    try (Graph g = new Graph()) {
      
      final float[][] A = new float[][]{
    	  { 1, 0, 0, 0, 0 },
    	  { 1, 2, 0, 0, 0 },
    	  { 1, 0, 3, 0, 0 },
    	  { 1, 0, 0, 4, 0 },
    	  { 1, 0, 0, 0, 5 }
    	}; 
      final float[][] b = new float[][] {
    	  {1}, {1}, {1}, {1}, {1}   
      };

      /*
      try (Tensor tA = Tensor.create(A); Tensor tb = Tensor.create(b)) {
          // The Java API doesn't yet include convenience functions for adding operations.
    	  g.opBuilder("matmul", "testOp").setAttr("dtype", A.dataType()).addInput(A).addInput(b).build();
      }

*/
      // Execute the "MyConst" operation in a Session.
      try (Session s = new Session(g); Tensor tA = Tensor.create(A);
           //Tensor output = s.runner().fetch("testOp").run().get(0)
        		   ) {
    	  g.opBuilder("MatMul", "megatest").setAttr("dtype", A.dataType());
        System.out.println(A[2][2]);
      }
      
    }
  }
}
