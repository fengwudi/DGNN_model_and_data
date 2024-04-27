// PartitionStrategy.java: interface defining a partition strategy

package partitioner;

import core.Edge;

import java.util.HashMap;
import java.util.List;

public interface PartitionStrategy {
    void performStep(Edge t, List<Integer> top_nodes, HashMap<Integer,Boolean> is_topk, PartitionState state);
}
