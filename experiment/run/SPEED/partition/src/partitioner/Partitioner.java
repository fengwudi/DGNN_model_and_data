// Partitioner.java: class that manage the partitioning procedure (multithread)

package partitioner;

import core.Edge;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import partitioner.coordinated_state.CoordinatedPartitionState;
import partitioner.strategies.*;
import application.Globals;

public class Partitioner {
    
    private List<Edge> dataset;
    private List<Integer> top_nodes;
    private HashMap<Integer,Boolean> is_topk;
    private PartitionStrategy algorithm;
    private Globals GLOBALS;

    public Partitioner(List<Edge> dataset, List<Integer> top_Nodes, HashMap<Integer,Boolean> is_topk, Globals G) {
        this.GLOBALS = G;
        this.dataset = dataset;
        this.top_nodes = top_Nodes;
        this.is_topk = is_topk;
        //"greedy", "hdrf", "hashing", "grid", "pds
        if (GLOBALS.PARTITION_STRATEGY.equalsIgnoreCase("hdrf")){ algorithm = new HDRF(GLOBALS); }
        else if (GLOBALS.PARTITION_STRATEGY.equalsIgnoreCase("hashing")){ algorithm = new Hashing(GLOBALS); }
        else if (GLOBALS.PARTITION_STRATEGY.equalsIgnoreCase("mymethod")){ algorithm = new MyMethod(GLOBALS); }
    }  
    
    public CoordinatedPartitionState performCoordinatedPartition(){
        return startCoordinated();
    }
    
    private CoordinatedPartitionState startCoordinated(){
        CoordinatedPartitionState state = new CoordinatedPartitionState(GLOBALS);
        int processors = GLOBALS.THREADS;
        ExecutorService executor=Executors.newFixedThreadPool(processors);
        int n = dataset.size();
        int subSize = n / processors + 1;
        for (int t = 0; t < processors; t++) {
            final int iStart = t * subSize;
            final int iEnd = Math.min((t + 1) * subSize, n);
            if (iEnd>=iStart){
                List<Edge> list= dataset.subList(iStart, iEnd);
                Runnable x = new PartitionerThread(list, top_nodes ,is_topk, state, algorithm, new LinkedList());
                executor.execute(x);
            }
        }
        try { 
            executor.shutdown();
            executor.awaitTermination(60, TimeUnit.DAYS);
        } catch (InterruptedException ex) {System.out.println("InterruptedException "+ex);ex.printStackTrace();}
        return state;
    }  
}
