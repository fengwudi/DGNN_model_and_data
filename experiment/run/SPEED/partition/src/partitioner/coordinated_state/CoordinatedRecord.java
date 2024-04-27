// CoordinatedRecord.java: class implementing the Record interface

package partitioner.coordinated_state;

import java.io.Serializable;
import java.util.Iterator;
import java.util.TreeSet;
import java.util.concurrent.atomic.AtomicBoolean;
import partitioner.Record;

public class CoordinatedRecord implements Serializable,Record{
    
    private TreeSet<Byte> partitions;   
    private AtomicBoolean lock;
    private int degree;
    private int importance;
    
    public CoordinatedRecord() {
        partitions = new TreeSet<Byte>();
        lock = new AtomicBoolean(true);
        degree = 0;
    }
    
    @Override
    public Iterator<Byte> getPartitions(){
        return partitions.iterator();
    }
    
    @Override
    public void addPartition(int m){
        if (m==-1){ System.out.println("ERRORE! record.addPartition(-1)"); System.exit(-1);}
        partitions.add( (byte) m);
    }
    
    public void addAll(TreeSet<Byte> tree){
        partitions.addAll(tree);
    }
    
    @Override
    public boolean hasReplicaInPartition(int m){
        return partitions.contains((byte) m);
    }
    
    @Override
    public synchronized boolean getLock(){
        return lock.compareAndSet(true, false);
    }
    
    @Override
    public synchronized boolean releaseLock(){
        return lock.compareAndSet(false, true);
    }
    
    @Override
    public int getReplicas(){
        return partitions.size();
    }

    @Override
    public int getDegree() {
        return degree;
    }

    @Override
    public double getImportance(){ return importance;}

    @Override
    public void incrementDegree() {
        this.degree++;
    }

    @Override
    public void incrementImportance(double ts,double maxtime){ this.importance += ts/maxtime;}
    
    public static TreeSet<Byte> intersection(CoordinatedRecord x, CoordinatedRecord y){
        TreeSet<Byte> result = (TreeSet<Byte>) x.partitions.clone();
        result.retainAll(y.partitions);
        return result;
    }    
    
}