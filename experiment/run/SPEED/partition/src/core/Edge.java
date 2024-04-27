// Edge.java: file defining the Edge object

package core;

import java.io.Serializable;

public class Edge implements Comparable,Serializable{
    private final int u;
    private final int v;
    private final double ts;

    public Edge(int u, int v, double ts) {
        this.u = u;
        this.v = v;
        this.ts = ts;
    }

    public int getU() {
        return u;
    }

    public int getV() {
        return v;
    }

    public double getTs() { return ts;}

    @Override
    public int hashCode() {
        String a = toString();
        int hash = a.hashCode();
        return hash;
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == null) {
            return false;
        }
        if (getClass() != obj.getClass()) {
            return false;
        }
        final Edge other = (Edge) obj;
        if (this.u != other.u) {
            if ((this.u == other.v)&&(this.v == other.u)){
                return true;
            }
            else return false;
        }
        if (this.v != other.v) {
            if ((this.u == other.v)&&(this.v == other.u)){
                return true;
            }
            else return false;
        }
        return true;
    }
    
    @Override
    public String toString(){
        String s = "";
        if (u<v){
            s = u+","+v;
        }
        else{
            s = v+","+u;
        }
        return s;
    }

    @Override
    public int compareTo(Object obj) {
        if (obj == null) {
            System.out.println("ERROR: Edge.compareTo -> obj == null");
            System.exit(-1);
        }
        if (getClass() != obj.getClass()) {
            System.out.println("ERROR: Edge.compareTo -> getClass() != obj.getClass()");
            System.exit(-1);
        }
        final Edge other = (Edge) obj;
        return this.toString().compareTo(obj.toString()); //lexicographic order
    }
}
