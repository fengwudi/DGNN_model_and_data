// Record.java: interface defining a record for each vertex

package partitioner;

import java.util.Iterator;

public interface Record {
    public Iterator<Byte> getPartitions();
    public void addPartition(int m);
    public boolean hasReplicaInPartition(int m);
    public boolean getLock();
    public boolean releaseLock();
    public int getReplicas();
    public int getDegree();
    public double getImportance();
    public void incrementDegree();
    public void incrementImportance(double ts,double maxtime);
}
