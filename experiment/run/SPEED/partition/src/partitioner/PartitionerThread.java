// PartitionerThread.java: thread that perform the partitioning

package partitioner;

import core.Edge;

import java.util.*;

public class PartitionerThread implements Runnable{

    private final List<Edge> list;
    private final List<Integer> top_nodes;
    private final HashMap<Integer,Boolean> is_topk;
    private final PartitionState state;
    private final PartitionStrategy algorithm;
    LinkedList<Integer> id_partitions;

    public PartitionerThread(List<Edge> list, List<Integer> top_nodes, HashMap<Integer,Boolean> is_topk, PartitionState state, PartitionStrategy algorithm, LinkedList<Integer> ids) {
        this.list = list;
        this.top_nodes = top_nodes;
        this.is_topk = is_topk;
        this.state = state;
        this.algorithm = algorithm;
        this.id_partitions = ids;
    }
    
    @Override
    public void run() {
        //Collections.reverse(list);
        List<Edge> train_list = list.subList(0, (int) (list.size() * 0.85));
        List<Edge> valid_test_list = list.subList((int)(list.size()*0.85),list.size());
        System.out.println(train_list.size());
        System.out.println(valid_test_list.size());
        Collections.reverse(train_list);
        ArrayList<Edge> list_new = new ArrayList<>();
        list_new.addAll(train_list);
        list_new.addAll(valid_test_list);
        System.out.println(list_new.size());
        //int i = 0;
        for (Edge t: list_new){
            algorithm.performStep(t, top_nodes, is_topk,state);
            //System.out.println(i);
            //i++;
        }
    }
}
