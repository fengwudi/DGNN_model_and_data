package application;

public class Globals {
    
    //CONSTANT
    public int SLEEP_LIMIT = 1024;
    public final static int PLACES = 4;
    public double K;  //control the number of republic nodes
    //APPLICATION PARAMETERS
    //MANDATORY
    public String INPUT_FILE_NAME;
    public int P;  //number of partitions
    //OPTIONAL
    public String PARTITION_STRATEGY= "hdrf"; // "hdrf", "random", "mymethod"
    public String DEGREE_COMPUTE = "normal";  // "normal", "decay"
    public double LAMBDA = 1;
    public double BETA = 0.1;
    public String DATASET;
    public int THREADS = Runtime.getRuntime().availableProcessors();
    public String OUTPUT_FILE_NAME;
    public double maxtime;

    public int SEED  = 0;
    
    public Globals(String[] args){
        parse_arguments(args);
    }
    
    private void parse_arguments(String[] args){
        try{
            DATASET = args[0];
            INPUT_FILE_NAME = "data/" + args[0] + ".txt";
            P = Integer.parseInt(args[1]);
            K = Double.parseDouble(args[2]);
            for(int i=3; i < args.length; i+=2){
                if(args[i].equalsIgnoreCase("-lambda")){
                    LAMBDA = Double.parseDouble(args[i+1]);
                }
                else if(args[i].equalsIgnoreCase("-seed")){
                    SEED = Integer.parseInt(args[i+1]);
                }
                else if(args[i].equalsIgnoreCase("-beta")){
                    BETA = Double.parseDouble(args[i+1]);
                }
                else if(args[i].equalsIgnoreCase("-degree_compute")){
                    DEGREE_COMPUTE = args[i+1];
                    if (DEGREE_COMPUTE.equalsIgnoreCase("normal")){}
                    else if(DEGREE_COMPUTE.equalsIgnoreCase("decay")){}
                    else{
                        System.out.println("\nInvalid degree compute method "+DEGREE_COMPUTE+". Aborting.");
                        System.out.println("Valid method: normal, decay.\n");
                        System.exit(-1);
                    }
                }
                else if(args[i].equalsIgnoreCase("-algorithm")){
                    PARTITION_STRATEGY =args[i+1];
                    if (PARTITION_STRATEGY.equalsIgnoreCase("greedy")){}
                    else if (PARTITION_STRATEGY.equalsIgnoreCase("hdrf")){}
                    else if (PARTITION_STRATEGY.equalsIgnoreCase("hashing")){}
                    else if (PARTITION_STRATEGY.equalsIgnoreCase("grid")){}
                    else if (PARTITION_STRATEGY.equalsIgnoreCase("pds")){}
                    else if (PARTITION_STRATEGY.equalsIgnoreCase("dbh")){}
                    else if (PARTITION_STRATEGY.equalsIgnoreCase("mymethod")){}
                    else{
                        System.out.println("\nInvalid algorithm "+PARTITION_STRATEGY+". Aborting.");
                        System.out.println("Valid algorithms: hdrf, random, mymethod.\n");
                        System.exit(-1);
                    }
                }
                else if(args[i].equalsIgnoreCase("-threads")){
                    THREADS = Integer.parseInt(args[i+1]);
                }
                else if(args[i].equalsIgnoreCase("-output")){
                    OUTPUT_FILE_NAME = args[i+1];
                }
                else throw new IllegalArgumentException();
            }
        } catch (Exception e){
            System.out.println("\nInvalid arguments ["+args.length+"]. Aborting.\n");
            System.out.println("Usage:\n VGP graphfile nparts [options]\n");
            System.out.println("Parameters:");
            System.out.println(" graphfile: the name of the file that stores the graph to be partitioned.");
            System.out.println(" nparts: the number of parts that the graph will be partitioned into. Maximum value 256.");
            System.out.println(" topk: the number of replicabel nodes ");
            System.out.println("\nOptions:");
            System.out.println(" -algorithm string");
            System.out.println("\t specifies the algorithm to be used (hdrf random mymethod). Default mymethod.");
            System.out.println(" -lambda double");
            System.out.println("\t specifies the lambda parameter for hdrf. Default 1.");
            System.out.println(" -threads integer");
            System.out.println("\t specifies the number of threads used by the application. Default all available processors.");
            System.out.println(" -output string");
            System.out.println("\t specifies the prefix for the name of the files where the output will be stored (files: prefix.info, prefix.edges and prefix.vertices).");
            System.out.println();
            System.exit(-1);
        }
    }
    
    public void print(){
        System.out.println("\tgraphfile: "+INPUT_FILE_NAME);
        System.out.println("\tparts: "+P);
        System.out.println("\ttopk: "+K);
        System.out.print("\talgorithm: "+PARTITION_STRATEGY);
        if (PARTITION_STRATEGY.equalsIgnoreCase("hdrf")){ System.out.println(" (lambda: "+LAMBDA+")"); }
        else System.out.println("");
        System.out.print("\tdegree_compute: "+DEGREE_COMPUTE);
        System.out.println("\tbeta: "+BETA);
        System.out.println("\tseed: "+SEED);
        System.out.println("\tthreads: "+THREADS);
        if (OUTPUT_FILE_NAME!=null){ System.out.println("\toutput: "+OUTPUT_FILE_NAME); }
    }
}
