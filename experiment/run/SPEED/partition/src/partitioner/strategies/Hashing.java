// Hashing.java: class implementing the Hashing partitioning algorithm


package partitioner.strategies;

import partitioner.coordinated_state.CoordinatedPartitionState;
import core.Edge;
import partitioner.PartitionState;
import partitioner.PartitionStrategy;
import partitioner.Record;
import application.Globals;

import java.util.HashMap;
import java.util.List;

public class Hashing implements PartitionStrategy{

    double seed;
    private Globals GLOBALS;
    
    public Hashing(Globals G) {
        seed = Math.random();
        this.GLOBALS = G;
    }
    
    @Override
    public void performStep(Edge e, List<Integer> top_nodes, HashMap<Integer,Boolean> is_topk, PartitionState state) {
        int P = GLOBALS.P;
        int u = e.getU();
        int v = e.getV();
        
        Record u_record = state.getRecord(u);
        Record v_record = state.getRecord(v);
        
        //*** ASK FOR LOCK
        int sleep = 2; while (!u_record.getLock()){ try{ Thread.sleep(sleep); }catch(Exception ex){} sleep = (int) Math.pow(sleep, 2);}
        sleep = 2; while (!v_record.getLock()){ try{ Thread.sleep(sleep); }catch(Exception ex){} sleep = (int) Math.pow(sleep, 2); 
        if (sleep>GLOBALS.SLEEP_LIMIT){u_record.releaseLock(); performStep(e,top_nodes,is_topk,state); return;} //TO AVOID DEADLOCK
        }
        //*** LOCK TAKEN
        
        int machine_id = Math.abs((int) ( (int) u*v*seed) % P);  
        
        //UPDATE EDGES
        state.incrementMachineLoad(machine_id,e);
        
        //UPDATE RECORDS
        if (state.getClass() == CoordinatedPartitionState.class){
            CoordinatedPartitionState cord_state = (CoordinatedPartitionState) state;
            //NEW UPDATE RECORDS RULE TO UPFDATE THE SIZE OF THE PARTITIONS EXPRESSED AS THE NUMBER OF VERTICES THEY CONTAINS
            if (!u_record.hasReplicaInPartition(machine_id)){ u_record.addPartition(machine_id); cord_state.incrementMachineLoadVertices(machine_id);}
            if (!v_record.hasReplicaInPartition(machine_id)){ v_record.addPartition(machine_id); cord_state.incrementMachineLoadVertices(machine_id);}
        }
        else{
            //1-UPDATE RECORDS
            if (!u_record.hasReplicaInPartition(machine_id)){ u_record.addPartition(machine_id);}
            if (!v_record.hasReplicaInPartition(machine_id)){ v_record.addPartition(machine_id);}
        }
          
        //*** RELEASE LOCK
        u_record.releaseLock();
        v_record.releaseLock();
    }
}
