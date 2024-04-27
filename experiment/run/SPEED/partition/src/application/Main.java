// Main.java: main file for VGP

package application;

import core.Edge;
import input.Input;

import java.io.IOException;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import output.Output;
import output.Statistics;
import partitioner.Partitioner;
import partitioner.coordinated_state.CoordinatedPartitionState;

public class Main {
    
    public static void main(String[] args) throws IOException {
        System.out.println("\n--------------------------------------------------");
        //System.out.println(" VGP: A Software Package for one-pass Vertex-cut balanced Graph Partitioning.");
        //System.out.println(" author: Yonxgxiang Liao");
        //System.out.println("--------------------------------------------------\n");
        Globals GLOBALS = new Globals(args);
        System.out.println(" Parameters:\n");
        GLOBALS.print();
        Statistics stat = new Statistics(GLOBALS);
        System.out.print("\n Loading graph into main memory... ");
        Input input = new Input(GLOBALS);
        List<Edge> x = input.getDataset();
        List<Integer> top_nodes = input.getTopKNodes();
        HashMap<Integer,Boolean> is_topk = input.get_is_topk();
        //for (int i = 0 ;i<GLOBALS.K; i++){
        //    System.out.println("\n" + top_nodes.get(i) +"yessssss\n");
        //}
        //System.out.println("\n" + top_nodes.get(0) +"yessssss\n");
        System.out.print("\n Running program... ");
        start(GLOBALS,stat,input,x,top_nodes,is_topk);
    }
    
    private static void start(Globals GLOBALS, Statistics stat, Input input, List<Edge> x, List<Integer> top, HashMap<Integer, Boolean> is_topk){
        long begin_time = System.currentTimeMillis();
        List<Edge> dataset = x;
        List<Integer> top_Nodes = top;
        //Collections.shuffle(dataset);
        Partitioner p = new Partitioner(dataset,top_Nodes,is_topk,GLOBALS);
        CoordinatedPartitionState state  = p.performCoordinatedPartition();
        long end_time = System.currentTimeMillis();
        long time = end_time-begin_time;
        //time /= 1000; //sec
        int [] load = state.getMachines_load();
        stat.computeReplicationFactor(state);  
        stat.computeStdDevLoad(load);
        double RF = round(stat.getReplicationFactor(),GLOBALS.PLACES);
        double std_dev = round(stat.getStdDevLoad(),GLOBALS.PLACES);  
        int [] load_edges = state.getMachines_load();
        int [] load_vertices = state.getMachines_loadVertices();
        int drop = state.getDropNum();
        int bigedge = state.getBigEdgeNum();
        int MAX_LOAD_EDGES = findMax(load_edges);
        int MAX_LOAD_VERTICES = findMax(load_vertices);
        System.out.println((int) time +" seconds");        
        System.out.println("\n Results:\n");
        System.out.println("\tReplication factor: "+RF);
        System.out.println("\tLoad relative standard deviation: "+std_dev);
        System.out.println("\tMax partition size (edge cardinality): "+MAX_LOAD_EDGES);
        System.out.println("\tMax partition size (vertex cardinality): "+MAX_LOAD_VERTICES);
        System.out.println("\t drop number: "+drop);
        System.out.println("\t edges between big nodes: "+bigedge);
        System.out.println("\n");     
        //WRITE OUTPUT ON FILE
        if (GLOBALS.OUTPUT_FILE_NAME!=null){
            //Output.writeInfo(GLOBALS, RF, std_dev, MAX_LOAD_EDGES, MAX_LOAD_VERTICES);
            //Output.writeVertexReplicas(GLOBALS, state);
            if(GLOBALS.PARTITION_STRATEGY.equalsIgnoreCase("hashing")){
                Output.writeVertexListHasing(GLOBALS, state, load_vertices);
            }
            else {
                Output.writeVertexListNew(GLOBALS, state, load_vertices);
            }
        }
    }
    
    public static int findMax(int [] array){
        int MAX = -1;
        for (int i =0; i<array.length; i++){
            if (array[i]>MAX){
                MAX = array[i];
            }
        }
        return MAX;
    }

    
    public static double round(double value, int places) {
        if (places < 0) throw new IllegalArgumentException();
        BigDecimal bd = new BigDecimal(value);
        bd = bd.setScale(places, RoundingMode.HALF_UP);
        return bd.doubleValue();
    }
}
