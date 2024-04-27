// Output.java: class to write the output partitions (info and vertices placement)

package output;

import application.Globals;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.SortedSet;
import partitioner.PartitionState;
import partitioner.Record;
import output.Statistics;
import partitioner.Partitioner;
import partitioner.coordinated_state.CoordinatedPartitionState;

import javax.xml.crypto.Data;

public class Output {
    
    public static void writeInfo(Globals GLOBALS, double RF, double std_dev, int MAX_LOAD_EDGES, int MAX_LOAD_VERTICES){
        DatWriter out = new DatWriter(GLOBALS.OUTPUT_FILE_NAME+".info");
        out.write("graphfile: "+GLOBALS.INPUT_FILE_NAME+"\n");
        out.write("parts: "+GLOBALS.P+"\n");
        out.write("algorithm: "+GLOBALS.PARTITION_STRATEGY);
        if (GLOBALS.PARTITION_STRATEGY.equalsIgnoreCase("hdrf")){ out.write(" (lambda: "+GLOBALS.LAMBDA+")\n"); }
        else out.write("\n");
        out.write("\n");
        out.write("Replication factor: "+RF+"\n");
        out.write("Load relative standard deviation: "+std_dev+"\n");
        out.write("Max partition size (edge cardinality): "+MAX_LOAD_EDGES+"\n");
        out.write("Max partition size (vertex cardinality): "+MAX_LOAD_VERTICES+"\n");
        out.close();        
    }
    
    public static void writeVertexReplicas(Globals GLOBALS, PartitionState state){
        DatWriter out = new DatWriter(GLOBALS.OUTPUT_FILE_NAME+".vertices");
        SortedSet<Integer> vertex_ids = state.getVertexIds();
        for (int x : vertex_ids){
            out.write(x+":");
            Record record = state.getRecord(x);
            Iterator<Byte> partitions =  record.getPartitions();
            while (partitions.hasNext()){
                int y = (( partitions.next()  & 0xFF ));                
                out.write(" "+y);
            }
            out.write("\n");
        }
        out.close();
    }

    public static void writeVertexList(Globals GLOBALS, PartitionState state, int [] vertex_Loads) {
        SortedSet<Integer> vertex_ids = state.getVertexIds();
        //int [] vertex_Loads = state.getMachines_loadVertices();
        List<Integer> shared_nodes = new ArrayList<>();
        //List<DatWriter> output_files = new ArrayList<DatWriter>();
        DatWriter[] output_files = new DatWriter[GLOBALS.P];
        DatWriter output_shared_nodes = null;
        //DatWriter dropped_info = new DatWriter("dropped_edges/" + "dropped_edge_info.txt");
        if (GLOBALS.PARTITION_STRATEGY.equalsIgnoreCase("hdrf")) {
            output_shared_nodes = new DatWriter("divided_nodes_seed_hdrf/" + GLOBALS.DATASET + "/" + GLOBALS.SEED + "/" + GLOBALS.DATASET + "_" + GLOBALS.P + "parts" + "/" + GLOBALS.OUTPUT_FILE_NAME + "shared" + ".txt");
            for (int i = 0; i < GLOBALS.P; i++) {
                output_files[i] = new DatWriter("divided_nodes_seed_hdrf/" + GLOBALS.DATASET + "/" + GLOBALS.SEED + "/" + GLOBALS.DATASET + "_" + GLOBALS.P + "parts" + "/" + GLOBALS.OUTPUT_FILE_NAME + String.valueOf(i) + ".txt");
            }
        }
        else if (GLOBALS.PARTITION_STRATEGY.equalsIgnoreCase("hashing")) {
            output_shared_nodes = new DatWriter("divided_nodes_seed_hashing/" + GLOBALS.DATASET + "/" + GLOBALS.SEED + "/" + GLOBALS.DATASET + "_" + GLOBALS.P + "parts" + "/" + GLOBALS.OUTPUT_FILE_NAME + "shared" + ".txt");
            for (int i = 0; i < GLOBALS.P; i++) {
                output_files[i] = new DatWriter("divided_nodes_seed_hashing/" + GLOBALS.DATASET + "/" + GLOBALS.SEED + "/" + GLOBALS.DATASET + "_" + GLOBALS.P + "parts" + "/" + GLOBALS.OUTPUT_FILE_NAME + String.valueOf(i) + ".txt");
            }
        }
        else {
            output_shared_nodes = new DatWriter("divided_nodes_seed/" + GLOBALS.DATASET + "/" + GLOBALS.SEED + "/" + GLOBALS.DATASET + "_" + GLOBALS.P + "parts_top" + (int) (GLOBALS.K * 100) + "/" + GLOBALS.OUTPUT_FILE_NAME + "shared" + ".txt");
            for (int i = 0; i < GLOBALS.P; i++) {
                output_files[i] = new DatWriter("divided_nodes_seed/" + GLOBALS.DATASET + "/" + GLOBALS.SEED + "/" + GLOBALS.DATASET + "_" + GLOBALS.P + "parts_top" + (int) (GLOBALS.K * 100) + "/" + GLOBALS.OUTPUT_FILE_NAME + String.valueOf(i) + ".txt");
            }
        }
        for (int x : vertex_ids) {
            Record record = state.getRecord(x);
            Iterator<Byte> partitions = record.getPartitions();
            int[] parts = new int[GLOBALS.P];
            int num_parts = 0;
            while (partitions.hasNext()) {
                int y = ((partitions.next() & 0xFF));
                parts[num_parts] = y;
                num_parts++;
            }
            //if (num_parts == GLOBALS.P){
            //    shared_nodes.add(x);
            //    for (int j = 0; j < GLOBALS.P; j++) {
            //        output_files[j].write(x + "\n");
            //    }
            //}
            if (num_parts == 1) {
                output_files[parts[0]].write(x + "\n");
            } else {
                int min_parts = findMinParts(parts, vertex_Loads, num_parts);
                //System.out.println(vertex_Loads[0]+"\n");
                output_files[min_parts].write(x + "\n");
                for (int j = 0; j < num_parts; j++) {
                    if (parts[j] != min_parts) {
                        vertex_Loads[parts[j]]--;
                    }
                }
            }
            for (int node : shared_nodes) {
                output_shared_nodes.write(node + "\n");
            }
            for (int i = 0; i < GLOBALS.P; i++) {
                output_files[i].close();
            }
            output_shared_nodes.close();
        }
    }

    public static void writeVertexListNew(Globals GLOBALS, PartitionState state, int [] vertex_Loads){
        SortedSet<Integer> vertex_ids = state.getVertexIds();
        //int [] vertex_Loads = state.getMachines_loadVertices();
        List<Integer> shared_nodes = new ArrayList<>();
        //List<DatWriter> output_files = new ArrayList<DatWriter>();
        DatWriter [] output_files = new DatWriter[GLOBALS.P];
        DatWriter output_shared_nodes = null;
        if (GLOBALS.PARTITION_STRATEGY.equalsIgnoreCase("hdrf")) {
            output_shared_nodes = new DatWriter("divided_nodes_seed_hdrf/" + GLOBALS.DATASET + "/" + GLOBALS.SEED + "/" + GLOBALS.DATASET + "_" + GLOBALS.P + "parts" + "/" + GLOBALS.OUTPUT_FILE_NAME + "shared" + ".txt");
            for (int i = 0; i < GLOBALS.P; i++) {
                output_files[i] = new DatWriter("divided_nodes_seed_hdrf/" + GLOBALS.DATASET + "/" + GLOBALS.SEED + "/" + GLOBALS.DATASET + "_" + GLOBALS.P + "parts" + "/" + GLOBALS.OUTPUT_FILE_NAME + String.valueOf(i) + ".txt");
            }
        }
        else if (GLOBALS.PARTITION_STRATEGY.equalsIgnoreCase("hashing")) {
            output_shared_nodes = new DatWriter("divided_nodes_seed_hashing/" + GLOBALS.DATASET + "/" + GLOBALS.SEED + "/" + GLOBALS.DATASET + "_" + GLOBALS.P + "parts" + "/" + GLOBALS.OUTPUT_FILE_NAME + "shared" + ".txt");
            for (int i = 0; i < GLOBALS.P; i++) {
                output_files[i] = new DatWriter("divided_nodes_seed_hashing/" + GLOBALS.DATASET + "/" + GLOBALS.SEED + "/" + GLOBALS.DATASET + "_" + GLOBALS.P + "parts" + "/" + GLOBALS.OUTPUT_FILE_NAME + String.valueOf(i) + ".txt");
            }
        }
        else {
            output_shared_nodes = new DatWriter("divided_nodes_seed/" + GLOBALS.DATASET + "/" + GLOBALS.SEED + "/" + GLOBALS.DATASET + "_" + GLOBALS.P + "parts_top" + (int) (GLOBALS.K * 100) + "/" + GLOBALS.OUTPUT_FILE_NAME + "shared" + ".txt");
            for (int i = 0; i < GLOBALS.P; i++) {
                output_files[i] = new DatWriter("divided_nodes_seed/" + GLOBALS.DATASET + "/" + GLOBALS.SEED + "/" + GLOBALS.DATASET + "_" + GLOBALS.P + "parts_top" + (int) (GLOBALS.K * 100) + "/" + GLOBALS.OUTPUT_FILE_NAME + String.valueOf(i) + ".txt");
            }
        }


        for (int x : vertex_ids){
            Record record = state.getRecord(x);
            Iterator<Byte> partitions =  record.getPartitions();
            int[] parts = new int[GLOBALS.P];
            int num_parts = 0;
            while (partitions.hasNext()){
                int y = (( partitions.next()  & 0xFF ));
                parts[num_parts] = y;
                num_parts++;
            }
            if (num_parts > 1){
                shared_nodes.add(x);
                for (int j = 0; j < GLOBALS.P; j++) {
                    output_files[j].write(x + "\n");
                }
            }
            else {
                output_files[parts[0]].write(x + "\n");
            }
        }
        for (int node : shared_nodes){
            output_shared_nodes.write(node + "\n");
        }
        for (int i = 0; i < GLOBALS.P; i++) {
            output_files[i].close();
        }
        output_shared_nodes.close();
    }

    public static void writeVertexListHasing(Globals GLOBALS, PartitionState state, int [] vertex_Loads){
        Random r = new Random(1);
        SortedSet<Integer> vertex_ids = state.getVertexIds();
        //int [] vertex_Loads = state.getMachines_loadVertices();
        //List<Integer> shared_nodes = new ArrayList<>();
        //List<DatWriter> output_files = new ArrayList<DatWriter>();
        DatWriter [] output_files = new DatWriter[GLOBALS.P];
        DatWriter output_shared_nodes = new DatWriter("divided_nodes_seed/"+ GLOBALS.DATASET + "/" + GLOBALS.SEED + "/" + GLOBALS.DATASET + "_" + GLOBALS.P + "parts_top" +(int)(GLOBALS.K*100) + "/" + GLOBALS.OUTPUT_FILE_NAME  + "shared" + ".txt");
        for (int i = 0; i < GLOBALS.P; i++) {
            output_files[i]= new DatWriter("divided_nodes_seed/"+ GLOBALS.DATASET + "/" + GLOBALS.SEED + "/" + GLOBALS.DATASET + "_" + GLOBALS.P + "parts_top" + (int)(GLOBALS.K*100) + "/" + GLOBALS.OUTPUT_FILE_NAME  + String.valueOf(i) + ".txt");
        }
        for (int x : vertex_ids) {
            int part = r.nextInt(GLOBALS.P) % GLOBALS.P;
            output_files[part].write(x + "\n");
        }
        for (int i = 0; i < GLOBALS.P; i++) {
            output_files[i].close();
        }
        output_shared_nodes.close();
    }

    public static int findMinParts(int [] parts, int [] vertexload,int num_parts){
        int MIN = Integer.MAX_VALUE;
        int min_parts = 0;
        for (int i =0; i<num_parts; i++){
            if (vertexload[parts[i]]<MIN){
                MIN = vertexload[parts[i]];
                min_parts = parts[i];
            }
        }
        return min_parts;
    }
}
