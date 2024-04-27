// Input.java: file to load the graph into main memory

package input;
import core.Edge;

import java.io.*;
import java.util.*;

import java.util.Comparator;
import application.Globals;
import partitioner.Record;

public class Input {
    
    private final Globals GLOBALS;
    private List<Edge> dataset;
    private List<Integer> top_nodes;
    private HashMap<Integer,Boolean> is_topk;
    private int vertices;
    private long edges;
    
    public Input(Globals G) throws IOException {
        this.GLOBALS = G;   
        edges = 0;
        vertices = 0;
        readDatasetFromFile();
    }
    
    private void readDatasetFromFile() throws IOException {
        long begin_time = System.currentTimeMillis();
        TreeSet<Integer> vertices_tree= new TreeSet<Integer>();
        //TreeSet<Edge> edges_tree = new TreeSet<Edge>();
        ArrayList<Integer> top = new ArrayList<Integer>();
        ArrayList<Edge> edges_tree = new ArrayList<Edge>();
        HashMap<Integer,Integer> degree = new HashMap<Integer,Integer>();
        HashMap<Integer,Double> importance = new HashMap<Integer, Double>();
        HashMap<Integer,Boolean> istopk = new HashMap<Integer, Boolean>();

        //读取最后时刻的时间戳
        StringBuilder builder = new StringBuilder();
        RandomAccessFile randomAccessFile = new RandomAccessFile(GLOBALS.INPUT_FILE_NAME, "r");
        long fileLastPointer = randomAccessFile.length() - 1;
        for (long filePointer = fileLastPointer; filePointer != -1; filePointer--) {
            randomAccessFile.seek(filePointer);
            int readByte = randomAccessFile.readByte();
            if (0xA == readByte) {
                if (filePointer == fileLastPointer) {
                    continue;
                }
                break;
            }
            if (0xD == readByte) {
                if (filePointer == fileLastPointer - 1) {
                    continue;
                }
                break;
            }
            builder.append((char) readByte);
        }

        //System.out.println(builder.reverse().toString());
        String lastline = builder.reverse().toString();
        String value[] = lastline.split(" ");
        double maxtime = Double.parseDouble(value[2]);
        GLOBALS.maxtime = maxtime;
        //System.out.println("\n maxtime="+maxtime);


        try {
            FileInputStream fis = new FileInputStream(new File(GLOBALS.INPUT_FILE_NAME));
            InputStreamReader isr = new InputStreamReader(fis);
            BufferedReader in = new BufferedReader(isr);
            String line;

            while((line = in.readLine())!=null){
                if (line.startsWith("#")){continue;} //skip comments
                String values[] = line.split(" ");
                int u = Integer.parseInt(values[0]);
                int v = Integer.parseInt(values[1]);
                double current_time = Double.parseDouble(values[2]);

                if (u!=v){  //self connection not allowed
                    Edge t = new Edge(u,v, current_time);
                    edges_tree.add(t);
                    edges++;
                    //System.out.println(t);

                    if( vertices_tree.add(u)){ vertices++; }
                    if( vertices_tree.add(v) ){ vertices++; }
                    
                    //DEGREE AND IMPORTANCE STATISTICS
                    if (!degree.containsKey(u)){ degree.put(u, 0); importance.put(u, 0.0);}
                    if (!degree.containsKey(v)){ degree.put(v, 0); importance.put(v, 0.0);}

                    int old_degree_u  = degree.get(u);
                    int old_degree_v  = degree.get(v);
                    degree.put(u, old_degree_u+1);
                    degree.put(v, old_degree_v+1);

                    double old_importance_u = importance.get(u);
                    double old_importance_v = importance.get(v);
                    //System.out.println(Math.exp(GLOBALS.BETA*(current_time-maxtime))+"\n");
                    //System.out.println(GLOBALS.BETA*(current_time/maxtime) + "\n");
                    //importance.put(u, old_importance_u + Math.exp(GLOBALS.BETA*(current_time-maxtime)));
                    importance.put(u, old_importance_u + GLOBALS.BETA*(current_time/maxtime));
                    importance.put(v, old_importance_v + GLOBALS.BETA*(current_time/maxtime));
                }
            }         
            in.close();
        } catch (IOException ex) {
            System.out.println("\nError: Input.readDatasetFromFile.\n\n");
            ex.printStackTrace();
            System.exit(-1);
        }          
        
        //DEBUG
        int MIN_DEGREE = Integer.MAX_VALUE;
        int MAX_DEGREE = Integer.MIN_VALUE;
        double MIN_IMPORTANCE = Double.MAX_VALUE;
        double MAX_IMPORTANCE = Double.MIN_VALUE;

        for (int v : degree.keySet()){
            int d = degree.get(v);
            if (d>MAX_DEGREE){ MAX_DEGREE = d; }
            if (d<MIN_DEGREE){ MIN_DEGREE = d; }
        }
        for (int v : importance.keySet()){
            double d = importance.get(v);
            if (d>MAX_IMPORTANCE){ MAX_IMPORTANCE = d; }
            if (d<MIN_IMPORTANCE){ MIN_IMPORTANCE = d; }
        }

        int topk = (int)(GLOBALS.K*vertices);
        System.out.println(vertices+'\n');
        System.out.println(topk+'\n');
        //find topk nodes
        if(GLOBALS.DEGREE_COMPUTE.equalsIgnoreCase("normal")){
            degree.entrySet();
            List<Map.Entry<Integer,Integer>> list = new ArrayList<>(degree.entrySet());
            Collections.sort(list, (o1, o2) -> (o2.getValue() - o1.getValue()));
            for (int i = 0; i < topk; i++){
                top.add(list.get(i).getKey());
                istopk.put(list.get(i).getKey(),true);
                //System.out.println(list.get(i).getKey() + ":" +degree.get(list.get(i).getKey()) + "\n");
            }
            for (int i = topk;i<vertices;i++){
                istopk.put(list.get(i).getKey(),false);
            }
            //System.out.println(top.get(1) +"yessssss\n");
        }
        else if (GLOBALS.DEGREE_COMPUTE.equalsIgnoreCase("decay")){
            importance.entrySet();
            List<Map.Entry<Integer,Double>> list = new ArrayList<>(importance.entrySet());
            Collections.sort(list, new Comparator<Map.Entry<Integer, Double>>(){
                @Override
                public int compare(Map.Entry<Integer, Double> o1, Map.Entry<Integer, Double> o2) {
                    if ((o2.getValue() - o1.getValue())>0)
                        return 1;
                    else if((o2.getValue() - o1.getValue())==0)
                        return 0;
                    else
                        return -1;
                }
            });
            for (int i = 0; i < topk; i++){
                top.add(list.get(i).getKey());
                istopk.put(list.get(i).getKey(),true);
                //System.out.println(list.get(i).getKey() + ":" + importance.get(list.get(i).getKey()) + "\n");
            }
            for (int i = topk;i<vertices;i++){
                istopk.put(list.get(i).getKey(),false);
            }
            //System.out.println(top.get(1) +"yessssss\n");
        }

        long end_time = System.currentTimeMillis();
        long time = end_time-begin_time;
        //time /= 1000; //sec
        System.out.println((int) time +" seconds");
        System.out.println("\n Info:\n");
        System.out.println("\tvertices: "+vertices);
        System.out.println("\tedges: "+edges);
        System.out.println("\tmin-degree: "+MIN_DEGREE);
        System.out.println("\tmax-degree: "+MAX_DEGREE);
        System.out.println("\tmin-importance: "+MIN_IMPORTANCE);
        System.out.println("\tmax-importance: "+MAX_IMPORTANCE);
        
        dataset = new ArrayList<Edge>(edges_tree);
        top_nodes = new ArrayList<Integer>(top);
        is_topk = new HashMap<Integer,Boolean>(istopk);
        vertices_tree.clear();
        edges_tree.clear();
        degree.clear();
    }
    
    public List<Edge> getDataset(){
        return dataset;
    }

    public int getVertices() {
        return vertices;
    }

    public long getEdges() {
        return edges;
    }

    public List<Integer> getTopKNodes() {
        return top_nodes;
    }

    public HashMap<Integer,Boolean> get_is_topk() {return is_topk;}
}