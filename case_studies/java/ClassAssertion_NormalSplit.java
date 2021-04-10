package ontology_embed;

import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.io.FileDocumentSource;
import org.semanticweb.owlapi.model.*;
import org.semanticweb.owlapi.reasoner.OWLReasoner;
import org.semanticweb.owlapi.reasoner.OWLReasonerFactory;
import org.semanticweb.owlapi.reasoner.structural.StructuralReasonerFactory;

import java.io.*;
import java.util.*;

public class ClassAssertion_NormalSplit {
    public static String onto_file = "data/helis_v1.00.origin.owl";
    public static String train_onto_file = "helis_normal_split/helis_v1.00.train.owl";
    public static String train_file = "helis_normal_split/train.csv";
    public static String valid_file = "valid.csv";
    public static String test_file = "test.csv";
    public static String class_file = "classes.txt";
    public static String individual_file = "individuals.txt";
    public static String inferred_class_file = "inferred_classes.txt";

    /**
     * Those declared (none-inferred) membership axioms are split into train (70%), valid (10%) and test (20%)
     * A training ontology is created for learning the embedding, by removing valid and test membership axioms
     * One negative sample is generated for each positive axiom in the training set 
     * @param args
     */
    public static void main(String[] args) throws OWLOntologyCreationException, IOException, OWLOntologyStorageException {
        if(args.length >= 2){
            onto_file = args[1]; train_onto_file = args[2];
            train_file = args[3]; valid_file = args[4]; test_file = args[5];
            class_file = args[6]; individual_file = args[7]; inferred_class_file = args[8];
        }

        OWLOntologyManager m = OWLManager.createOWLOntologyManager();
        OWLOntology o = m.loadOntologyFromOntologyDocument(new FileDocumentSource(new File(onto_file)));
        OWLReasonerFactory reasonerFactory = new StructuralReasonerFactory();
        OWLReasoner reasoner = reasonerFactory.createReasoner(o);
        Set<OWLNamedIndividual> inds = o.getIndividualsInSignature();


        // get and save all individuals and classes
        ArrayList<String> classes_str = new ArrayList<>();
        ArrayList<String> individuals_str = new ArrayList<>();
        for (OWLClass c : o.getClassesInSignature()) {
            String c_str = c.toString();
            if (!classes_str.contains(c_str) && !c_str.equals("<http://www.w3.org/2002/07/owl#Thing>")) {
                classes_str.add(c_str);
            }
        }
        for (OWLNamedIndividual i : o.getIndividualsInSignature()) {
            String i_str = i.toString();
            if (!individuals_str.contains(i_str)) {
                individuals_str.add(i_str);
            }
        }
        save_classes_individuals(classes_str, individuals_str, class_file, individual_file);
        System.out.println("classes: " + classes_str.size() + ", individuals: " + individuals_str.size());

        // get all individual typing axioms (samples)
        ArrayList<Map.Entry<OWLNamedIndividual, OWLClass>> typing_samples = new ArrayList<>();
        ArrayList<OWLClass> classes = new ArrayList<>();
        for (OWLNamedIndividual i : inds){
            for(OWLClass c : reasoner.getTypes(i, true).getFlattened()){
                Map.Entry<OWLNamedIndividual, OWLClass> ent = new AbstractMap.SimpleEntry<>(i, c);
                typing_samples.add(ent);
                if(!classes.contains(c)){
                    classes.add(c);
                }
            }
        }

        // Split the samples into train, valid and test
        System.out.println("Class assertions #: " + typing_samples.size());
        Collections.shuffle(typing_samples);
        int num = typing_samples.size();
        int index1 = (int) ( num * 0.7);
        int index2 = index1 + (int) (num * 0.1);
        ArrayList<Map.Entry<OWLNamedIndividual, OWLClass>> typing_samples_train = new ArrayList<>(typing_samples.subList(0, index1));
        ArrayList<Map.Entry<OWLNamedIndividual, OWLClass>> typing_samples_valid = new ArrayList<>(typing_samples.subList(index1, index2));
        ArrayList<Map.Entry<OWLNamedIndividual, OWLClass>> typing_samples_test = new ArrayList<>(typing_samples.subList(index2, num));
        System.out.println(("train (positive): " + typing_samples_train.size() + ", valid: " + typing_samples_valid.size() +
                ", test: " + typing_samples_test.size()));

        // Get the negative training samples
        ArrayList<Map.Entry<OWLNamedIndividual, OWLClass>> typing_samples_train_neg =
                getNegativeSamples(typing_samples_train, reasoner, classes);

        // Save samples
        ArrayList<String> train = sample_to_string(typing_samples_train, "1");
        ArrayList<String> neg_train = sample_to_string(typing_samples_train_neg, "0");
        train.addAll(neg_train);
        ArrayList<String> valid = sample_to_string(typing_samples_valid, null);
        ArrayList<String> test = sample_to_string(typing_samples_test, null);
        save_sample(train_file, train);
        save_sample(valid_file, valid);
        save_sample(test_file, test);

        // Save inferred classes of individuals
        HashMap<OWLNamedIndividual, ArrayList<OWLClass>> ind_classes = get_inferred_classes(reasoner, inds);
        save_individual_classes(ind_classes, inferred_class_file);

        // Remove the test and valid typing axioms from the original ontology
        export_train_ontology(typing_samples_valid, typing_samples_test, o, m, train_onto_file);

    }

    static void export_train_ontology(ArrayList<Map.Entry<OWLNamedIndividual, OWLClass>> typing_samples_valid,
                                   ArrayList<Map.Entry<OWLNamedIndividual, OWLClass>> typing_samples_test,
                                   OWLOntology o, OWLOntologyManager m, String out_file) throws FileNotFoundException, OWLOntologyStorageException {
        OWLDataFactory df = OWLManager.getOWLDataFactory();

        ArrayList<Map.Entry<OWLNamedIndividual, OWLClass>> typing_remove = new ArrayList<>(typing_samples_valid);
        typing_remove.addAll(typing_samples_test);
        for (Map.Entry<OWLNamedIndividual, OWLClass> ent : typing_remove){
            OWLClassAssertionAxiom caa = df.getOWLClassAssertionAxiom(ent.getValue(), ent.getKey());
            RemoveAxiom ra = new RemoveAxiom(o, caa);
            List<RemoveAxiom> ral = Collections.singletonList(ra);
            m.applyChanges(ral);
        }
        m.saveOntology(o, new FileOutputStream(new File(out_file)));
    }

    static HashMap<OWLNamedIndividual, ArrayList<OWLClass>> get_inferred_classes(OWLReasoner reasoner, Set<OWLNamedIndividual> inds){
        HashMap<OWLNamedIndividual, ArrayList<OWLClass>> ind_classes = new HashMap<>();
        for (OWLNamedIndividual i : inds){
            ArrayList<OWLClass> inferred_classes = new ArrayList<>();
            Set<OWLClass> tmp_classes = reasoner.getTypes(i, true).getFlattened();
            for(OWLClass c : reasoner.getTypes(i, false).getFlattened()){
                if(!tmp_classes.contains(c)){
                    inferred_classes.add(c);
                }
            }
            ind_classes.put(i, inferred_classes);
        }
        return ind_classes;
    }

    static ArrayList<String> sample_to_string(ArrayList<Map.Entry<OWLNamedIndividual, OWLClass>> samples, String tag){
        ArrayList<String> samples_str = new ArrayList<>();
        for(Map.Entry<OWLNamedIndividual, OWLClass> s: samples){
            OWLNamedIndividual i = s.getKey();
            OWLClass c = s.getValue();
            if(tag == null) {
                String str = i.toString() + "," + c.toString();
                samples_str.add(str);
            }else{
                String str = i.toString() + "," + c.toString() + "," + tag;
                samples_str.add(str);
            }
        }
        return samples_str;
    }

    static void save_sample(String file_name, ArrayList<String> samples) throws IOException {
        File fout = new File(file_name);
        FileOutputStream fos = new FileOutputStream(fout);
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fos));
        for(String s: samples){
            bw.write(s + "\n");
        }
        bw.close();
    }

    static ArrayList<Map.Entry<OWLNamedIndividual, OWLClass>> getNegativeSamples(
            ArrayList<Map.Entry<OWLNamedIndividual, OWLClass>> pos_samples, OWLReasoner reasoner,
            ArrayList<OWLClass> classes){
        ArrayList<Map.Entry<OWLNamedIndividual, OWLClass>> neg_samples = new ArrayList<>();
        Random rand = new Random();
        for(Map.Entry<OWLNamedIndividual, OWLClass> pos_sample : pos_samples){
            OWLNamedIndividual i = pos_sample.getKey();
            OWLClass c = pos_sample.getValue();
            ArrayList<OWLClass> tmp_classes = new ArrayList<>(classes);
            tmp_classes.remove(c);
            for(OWLClass i_c : reasoner.getTypes(i, false).getFlattened()){
                tmp_classes.remove(i_c);
            }
            OWLClass neg_c = tmp_classes.get(rand.nextInt(tmp_classes.size()));
            Map.Entry<OWLNamedIndividual, OWLClass> neg_sample = new AbstractMap.SimpleEntry<>(i, neg_c);
            neg_samples.add(neg_sample);
        }
        return neg_samples;
    }

    static void save_classes_individuals(ArrayList<String> classes, ArrayList<String> individuals,
                                         String class_file, String individual_file) throws IOException {

        FileWriter cw = new FileWriter(class_file);
        BufferedWriter cbw = new BufferedWriter(cw);
        PrintWriter cout = new PrintWriter(cbw);
        for (String s : classes) {
            cout.println(s);
        }
        cbw.close();
        cw.close();

        FileWriter iw = new FileWriter(individual_file);
        BufferedWriter ibw = new BufferedWriter(iw);
        PrintWriter iout = new PrintWriter(ibw);
        for (String i : individuals){
            iout.println(i);
        }
        ibw.close();
        iw.close();
    }

    static void save_individual_classes(HashMap<OWLNamedIndividual, ArrayList<OWLClass>> ind_classes, String file) throws IOException {
        File fout = new File(file);
        FileOutputStream fos = new FileOutputStream(fout);
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fos));
        for(OWLNamedIndividual ind : ind_classes.keySet()){
            String s = ind.toString();
            for(OWLClass c: ind_classes.get(ind)){
                s = s + ',' + c.toString();
            }
            bw.write(s + "\n");
        }
        bw.close();
    }
}
