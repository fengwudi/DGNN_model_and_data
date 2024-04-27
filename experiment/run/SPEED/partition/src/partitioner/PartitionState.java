// PartitionState.java: interface defining a partition state

package partitioner;

import core.Edge;

import java.util.SortedSet;

public interface PartitionState {
    public Record getRecord(int x);
    public int getMachineLoad(int m);
    public void incrementMachineLoad(int m, Edge e);
    public int getMinLoad();
    public int getMaxLoad();
    public void writedropinfo(int u, int v, double ts, int partu, int partv);
    public int getDropNum();
    public int getBigEdgeNum();
    public void incrementDropNum();
    public void incrementBigEdges();
    public int[] getMachines_load();
    public int getTotalReplicas();
    public int getNumVertices();
    public SortedSet<Integer> getVertexIds();

}
