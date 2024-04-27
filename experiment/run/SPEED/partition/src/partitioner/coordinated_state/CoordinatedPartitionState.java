// CoordinatedPartitionState.java: class implementing the PartitionState interface

package partitioner.coordinated_state;

import java.util.HashMap;
import java.util.concurrent.atomic.AtomicInteger;
import partitioner.PartitionState;
import application.Globals;
import core.Edge;
import java.util.SortedSet;
import java.util.TreeSet;
import output.DatWriter;

public class CoordinatedPartitionState implements PartitionState{
    private HashMap<Integer,CoordinatedRecord> record_map;
    private AtomicInteger[] machines_load_edges;
    private AtomicInteger[] machines_load_vertices;
    private final Globals GLOBALS; 
    int MAX_LOAD;
    int DROP;
    int BIG_EDGE; //Edges between big nodes
    //DatWriter out; //to print the final partition of each edge
    private DatWriter dropped_info;
    private DatWriter[] output_files;

    public CoordinatedPartitionState(Globals G) {
        this.GLOBALS = G;
        record_map = new HashMap<Integer,CoordinatedRecord>();
        machines_load_edges = new AtomicInteger[GLOBALS.P];
        for (int i = 0; i<machines_load_edges.length;i++){ 
            machines_load_edges[i] = new AtomicInteger(0); 
        }        
        machines_load_vertices = new AtomicInteger[GLOBALS.P];
        for (int i = 0; i<machines_load_vertices.length;i++){ 
            machines_load_vertices[i] = new AtomicInteger(0); 
        }        
        MAX_LOAD = 0;
        DROP = 0;

        dropped_info = new DatWriter("divided_nodes_seed/" + GLOBALS.DATASET + "/" + GLOBALS.SEED + "/" + GLOBALS.DATASET + "_" + GLOBALS.P + "parts_top" + (int) (GLOBALS.K * 100) + "/" + "dropped_edge_info.txt");
        //output_files = new DatWriter[GLOBALS.P];
        //for (int i = 0; i < GLOBALS.P; i++) {
        //    output_files[i] = new DatWriter("dropped_edges/" + "outbound" + String.valueOf(i) + ".txt");
        //}
        //if (GLOBALS.OUTPUT_FILE_NAME!=null){
       //     out = new DatWriter(GLOBALS.OUTPUT_FILE_NAME+".edges");
        //}
    }
    
    public synchronized void incrementMachineLoadVertices(int m) {
        machines_load_vertices[m].incrementAndGet();
    }
    
    public int[] getMachines_loadVertices() {
        int [] result = new int[machines_load_vertices.length];
        for (int i = 0; i<machines_load_vertices.length;i++){ 
            result[i] = machines_load_vertices[i].get();
        }
        return result;
    }

    @Override
    public synchronized CoordinatedRecord getRecord(int x){
        if (!record_map.containsKey(x)){
            record_map.put(x, new CoordinatedRecord());
        }
        return record_map.get(x);
    }
    
    @Override
    public int getNumVertices(){
        return record_map.size();
    }
    
    @Override
     public int getTotalReplicas(){
        int result = 0;
        for (int x : record_map.keySet()){
            int r = record_map.get(x).getReplicas();
            if (r>0){
                result += record_map.get(x).getReplicas();
            }
            else{
                result++;
            }
        }
        return result;
    }

    @Override
    public synchronized int getMachineLoad(int m) {
        return machines_load_edges[m].get();
    }

    @Override
    public synchronized void incrementMachineLoad(int m, Edge e) {
        int new_value = machines_load_edges[m].incrementAndGet();
        if (new_value>MAX_LOAD){
            MAX_LOAD = new_value;
        }
       // if (GLOBALS.OUTPUT_FILE_NAME!=null){
        //    out.write(e+": "+m+"\n");
        //}
    }

    @Override
    public synchronized void writedropinfo(int u, int v, double ts, int partu, int partv) {
        dropped_info.write(u + " " + v + " " + ts + " " + partu + " " + partv + "\n");
        //int part_u = ((partu & 0xFF));
        //int part_v = ((partv & 0xFF));
        //System.out.println(u);
        //output_files[part_u].write(v + "\n");
        //output_files[part_v].write(u + "\n");
        // if (GLOBALS.OUTPUT_FILE_NAME!=null){
        //    out.write(e+": "+m+"\n");
        //}
    }

    @Override
    public synchronized int getDropNum(){return DROP;}

    @Override
    public synchronized int getBigEdgeNum(){return BIG_EDGE;}

    @Override
    public synchronized void incrementDropNum(){
        DROP++;
    }

    @Override
    public synchronized void incrementBigEdges(){BIG_EDGE++;}

    @Override
    public int[] getMachines_load() {
        int [] result = new int[machines_load_edges.length];
        for (int i = 0; i<machines_load_edges.length;i++){ 
            result[i] = machines_load_edges[i].get();
        }
        return result;
    }

    @Override
    public synchronized int getMinLoad() {
        int MIN_LOAD = Integer.MAX_VALUE;
        for (AtomicInteger load : machines_load_edges) {
            int loadi = load.get();
            if (loadi<MIN_LOAD){
                MIN_LOAD = loadi;
            }
        }
        return MIN_LOAD;
    }

    @Override
    public int getMaxLoad() {
        return MAX_LOAD;
    }

    @Override
    public SortedSet<Integer> getVertexIds() {
        //if (GLOBALS.OUTPUT_FILE_NAME!=null){ out.close(); }
        return new TreeSet<Integer>(record_map.keySet());
    }
    
}
