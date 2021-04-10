package ontology_embed;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import uio.ifi.ontology.toolkit.projection.controller.GraphProjectionManager;
import uio.ifi.ontology.toolkit.projection.controller.triplestore.RDFoxProjectionManager;
import uio.ifi.ontology.toolkit.projection.controller.triplestore.RDFoxSessionManager;

public class Ontology_Projector {

    private static String filePath = "file:/dir/go.train.owl";
    private static String outFile = "go.train.projection.ttl"; //only a relative path can be used here; no absolute path
    // use .ttl or manually transform it from .owl to ttl if it is input into OWL2Vec_Plus.py

    public static void main(String [] args) throws Exception {
        RDFoxSessionManager man = new RDFoxSessionManager();
        man.createNewSessionForEmbeddings(filePath);

        GraphProjectionManager pMan = man.getSession(filePath);
        RDFoxProjectionManager rMan;
        if (pMan instanceof RDFoxProjectionManager) {
            rMan = (RDFoxProjectionManager) man.getSession(filePath);
            rMan.exportMaterielizationSnapshot(outFile);
//            cleanSavedModel(outFile);
        }
        System.out.println("Saved projection as: " + outFile);
    }

    private static void cleanSavedModel(String outFile) {
        try {
            String tmpFile = "tmp.txt";
            File out = new File(outFile);
            File tmp = new File(tmpFile);
            BufferedReader reader = new BufferedReader(new FileReader(out));
            BufferedWriter writer = new BufferedWriter(new FileWriter(tmp));
            String line = null;
            int numLines = 0;
            int eliminatedLines = 0;
            while ((line = reader.readLine()) != null) {
                if (line.startsWith("\"") && !(line.endsWith(";"))) {
                    String toEliminate = line;
                    while (toEliminate != null) {
                        eliminatedLines++;
                        if (toEliminate.endsWith("\" .")) {
                            break;
                        }
                        System.out.println("eliminated: " + toEliminate);
                        toEliminate = reader.readLine();
                    }
                } else {
                    numLines++;
                    writer.write(line + "\n");
                }
            }
            writer.flush();
            writer.close();
            tmp.renameTo(out);
            System.out.println("Written " + numLines + " lines");
            System.out.println("Eliminated " + eliminatedLines + " lines");
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

}
