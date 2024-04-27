// DatWriter.java: object to easily write a simple file .dat

package output;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;

public class DatWriter {
    
    private String FILE_NAME;
    private BufferedWriter bw;
    
    public DatWriter (String f){
        FILE_NAME = f;
        open();
    }
    
    private void open(){
        try{
            File file = new File(FILE_NAME);
            // if file doesnt exists, then create it
            File parent = new File(file.getParent());
            if (!parent.exists()) {
                parent.mkdirs();
            }
            file.createNewFile();
            FileWriter fw = new FileWriter(file.getAbsoluteFile());
            bw = new BufferedWriter(fw);
        }catch(Exception e){
            System.out.println("ERRORE DatWriter.open() "+FILE_NAME);
            e.printStackTrace();
            System.exit(-1);
        }
    } 
    
    public void write(String content){
        try{
            bw.write(content);
        }
        catch(Exception e){
            System.out.println("ERRORE DatWriter.write("+content+") "+FILE_NAME);
            e.printStackTrace();
            System.exit(-1);
        }
    }
    
    public void close(){
       try{
            bw.close();
        }
        catch(Exception e){
            System.out.println("ERRORE DatWriter.close() "+FILE_NAME);
            e.printStackTrace();
            System.exit(-1);
        } 
    }
    
}
