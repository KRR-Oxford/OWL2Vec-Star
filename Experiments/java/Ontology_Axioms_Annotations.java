package ontology_embed;

import org.semanticweb.elk.owlapi.ElkReasonerFactory;
import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.reasoner.*;
import org.semanticweb.owlapi.model.*;
import org.semanticweb.owlapi.io.*;
import org.semanticweb.owlapi.search.EntitySearcher;
import org.semanticweb.owlapi.util.*;
import org.semanticweb.owlapi.manchestersyntax.renderer.*;

import java.io.*;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;

import org.semanticweb.owlapi.util.InferredAxiomGenerator;
import org.semanticweb.owlapi.util.InferredOntologyGenerator;
import org.semanticweb.owlapi.util.InferredSubClassAxiomGenerator;
import org.semanticweb.owlapi.util.InferredEquivalentClassAxiomGenerator;
import org.semanticweb.owlapi.reasoner.OWLReasoner;
import org.semanticweb.HermiT.Reasoner;
import org.semanticweb.owlapi.reasoner.OWLReasonerConfiguration;

public class Ontology_Axioms_Annotations {
    private static String prefreasoner = "none"; //hermit, elk, none
    private static String onto_file_name = "foodon-merged.train.owl";
    private static String axiom_file = "axioms.txt";
    private static boolean extract_annotation = true; // true or false
    private static String annotation_file = "annotations.txt";

    public static class SimpleShortFormProvider1 implements ShortFormProvider, Serializable {
        @Override
        public String getShortForm(OWLEntity entity) {
            return entity.getIRI().toString();
        }

        public void dispose() {
            ;
        }
    }

    public static void main(String[] args) throws OWLOntologyCreationException, IOException {
        if (args.length >= 2) {
            prefreasoner = args[1];
            onto_file_name = args[2];
            axiom_file = args[3];
            annotation_file = args[4];
        }

        OWLOntologyManager manager = OWLManager.createOWLOntologyManager();
        OWLOntology ont = manager.loadOntologyFromOntologyDocument(new FileDocumentSource(new File(onto_file_name)));

        System.out.println("Extract axioms ...");
        if (prefreasoner.equals("elk")) {
            axiom_corpus_elk(ont);
        } else if (prefreasoner.equals("hermit")) {
            axiom_corpus_hermit(manager, ont);
        } else if (prefreasoner.equals("none")) {
            axiom_corpus_no_inference(ont);
        } else {
            System.out.println(prefreasoner + " not implemented");
        }

        if(extract_annotation) {
            System.out.println("Extract annotations ...");
            extract_annotations(ont);
        }
    }

    private static void axiom_corpus_no_inference(OWLOntology ont) throws IOException {

        OWLObjectRenderer renderer = new ManchesterOWLSyntaxOWLObjectRendererImpl();
        renderer.setShortFormProvider(new SimpleShortFormProvider1());
        ArrayList<String> axiom_sentences = new ArrayList<>();

        // classes and their axioms
        Set<OWLClass> classes = ont.getClassesInSignature();
        classes.addAll(ont.getClassesInSignature());
        for (OWLClass c : classes) {
            // axiom_sentences.add(renderer.render(c));
            Set<OWLClassAxiom> ontoaxioms = ont.getAxioms(c);
            for (OWLClassAxiom c_axiom : ontoaxioms) {
                String c_axiom_str = renderer.render(c_axiom);
                c_axiom_str = c_axiom_str.replaceAll("\n", " ").replaceAll(",", " ").replaceAll("\\)", " ").replaceAll("\\(", " ");
                c_axiom_str = remove_datatype(c_axiom_str);
                if (!axiom_sentences.contains(c_axiom_str)) {
                    axiom_sentences.add(c_axiom_str);
                }
            }
        }
        System.out.println("classes done");

        // individuals and their axioms
        Set<OWLNamedIndividual> inds = ont.getIndividualsInSignature();
        inds.addAll(ont.getIndividualsInSignature());
        for (OWLNamedIndividual i : inds) {
            // axiom_sentences.add(renderer.render(i));
            Set<OWLIndividualAxiom> i_axioms = ont.getAxioms(i);
            for (OWLIndividualAxiom i_axiom : i_axioms) {
                String i_axiom_str = renderer.render(i_axiom);
                i_axiom_str = i_axiom_str.replaceAll("\n", " ").replaceAll(",", " ").replaceAll("\\)", " ").replaceAll("\\(", " ");
                i_axiom_str = remove_datatype(i_axiom_str);
                if (!axiom_sentences.contains(i_axiom_str)) {
                    axiom_sentences.add(i_axiom_str);
                }
            }
        }
        System.out.println("individuals done");

        save_axioms(axiom_sentences);
        System.out.println("axioms: " + axiom_sentences.size());
    }


    private static void axiom_corpus_elk(OWLOntology ont) throws OWLOntologyCreationException, IOException {
        OWLOntologyManager outputManager = OWLManager.createOWLOntologyManager();
        ConsoleProgressMonitor progressMonitor = new ConsoleProgressMonitor();
        OWLReasonerConfiguration config = new SimpleConfiguration(progressMonitor);
        ElkReasonerFactory f1 = new ElkReasonerFactory();
        OWLReasoner reasoner = f1.createReasoner(ont, config);
        reasoner.precomputeInferences();

        List<InferredAxiomGenerator<? extends OWLAxiom>> gens = new ArrayList<>();
        gens.add(new InferredSubClassAxiomGenerator());
        gens.add(new InferredEquivalentClassAxiomGenerator());
        gens.add(new InferredClassAssertionAxiomGenerator());
        OWLOntology infOnt = outputManager.createOntology();
        InferredOntologyGenerator iog = new InferredOntologyGenerator(reasoner, gens);
        iog.fillOntology(outputManager.getOWLDataFactory(), infOnt);

        OWLObjectRenderer renderer = new ManchesterOWLSyntaxOWLObjectRendererImpl();
        renderer.setShortFormProvider(new SimpleShortFormProvider1());
        ArrayList<String> axiom_sentences = new ArrayList<>();

        // classes and their axioms
        Set<OWLClass> classes = infOnt.getClassesInSignature();
        classes.addAll(ont.getClassesInSignature());
        for (OWLClass c : classes) {
            // axiom_sentences.add(renderer.render(c));
            Set<OWLClassAxiom> ontoaxioms = infOnt.getAxioms(c);
            ontoaxioms.addAll(ont.getAxioms(c));
            for (OWLClassAxiom c_axiom : ontoaxioms) {
                String c_axiom_str = renderer.render(c_axiom);
                c_axiom_str = c_axiom_str.replaceAll("\n", " ").replaceAll(",", " ").replaceAll("\\)", " ").replaceAll("\\(", " ");
                c_axiom_str = remove_datatype(c_axiom_str);
                if (!axiom_sentences.contains(c_axiom_str)) {
                    axiom_sentences.add(c_axiom_str);
                }
            }
        }
        System.out.println("classes done");

        // individuals and their axioms
        Set<OWLNamedIndividual> inds = infOnt.getIndividualsInSignature();
        inds.addAll(ont.getIndividualsInSignature());
        for (OWLNamedIndividual i : inds) {
            // axiom_sentences.add(renderer.render(i));
            Set<OWLIndividualAxiom> i_axioms = infOnt.getAxioms(i);
            i_axioms.addAll(ont.getAxioms(i));
            for (OWLIndividualAxiom i_axiom : i_axioms) {
                String i_axiom_str = renderer.render(i_axiom);
                i_axiom_str = i_axiom_str.replaceAll("\n", " ").replaceAll(",", " ").replaceAll("\\)", " ").replaceAll("\\(", " ");
                i_axiom_str = remove_datatype(i_axiom_str);
                if (!axiom_sentences.contains(i_axiom_str)) {
                    axiom_sentences.add(i_axiom_str);
                }
            }
        }
        System.out.println("individuals done");

        save_axioms(axiom_sentences);
        System.out.println("axioms: " + axiom_sentences.size());
    }


    private static void axiom_corpus_hermit(OWLOntologyManager manager, OWLOntology ont) throws IOException {
        OWLReasonerFactory reasonerFactory = new Reasoner.ReasonerFactory();
        OWLReasoner reasoner = reasonerFactory.createReasoner(ont);
        OWLDataFactory factory = manager.getOWLDataFactory();
        reasoner.precomputeInferences();
        InferredSubClassAxiomGenerator generator = new InferredSubClassAxiomGenerator();
        Set<OWLSubClassOfAxiom> axioms = generator.createAxioms(factory, reasoner);
        manager.addAxioms(ont, axioms);
        OWLObjectRenderer renderer = new ManchesterOWLSyntaxOWLObjectRendererImpl();
        renderer.setShortFormProvider(new SimpleShortFormProvider1());

        ArrayList<String> axiom_sentences = new ArrayList<>();

        Set<OWLClass> classes = ont.getClassesInSignature();
        for (OWLClass c : classes) {
            // axiom_sentences.add(renderer.render(c));
            Set<OWLClassAxiom> c_axioms = ont.getAxioms(c);
            for (OWLClassAxiom c_axiom : c_axioms) {
                String c_axiom_str = renderer.render(c_axiom);
                c_axiom_str = c_axiom_str.replaceAll("\n", " ").replaceAll(",", " ").replaceAll("\\)", " ").replaceAll("\\(", " ");
                c_axiom_str = remove_datatype(c_axiom_str);
                if (!axiom_sentences.contains(c_axiom_str)) {
                    axiom_sentences.add(c_axiom_str);
                }
            }
        }
        System.out.println("classes done");

        Set<OWLNamedIndividual> inds = ont.getIndividualsInSignature();
        for (OWLNamedIndividual i : inds) {
            // axiom_sentences.add(renderer.render(i));
            Set<OWLIndividualAxiom> i_axioms = ont.getAxioms(i);
            for (OWLIndividualAxiom i_axiom : i_axioms) {
                String i_axiom_str = renderer.render(i_axiom);
                i_axiom_str = i_axiom_str.replaceAll("\n", " ").replaceAll(",", " ").replaceAll("\\)", " ").replaceAll("\\(", " ");
                i_axiom_str = remove_datatype(i_axiom_str);
                if (!axiom_sentences.contains(i_axiom_str)) {
                    axiom_sentences.add(i_axiom_str);
                }
            }
        }
        System.out.println("individuals done");

        save_axioms(axiom_sentences);
        System.out.println("axioms: " + axiom_sentences.size());

    }

    private static void extract_annotations(OWLOntology ont) throws IOException {
        ArrayList<String> ann_axioms = new ArrayList<>();

        for (OWLEntity e : ont.getSignature()) {
            for (OWLAnnotation a : EntitySearcher.getAnnotations(e, ont)) {
                OWLAnnotationProperty prop = a.getProperty();
                OWLAnnotationValue val = a.getValue();
                if (val instanceof OWLLiteral) {
                    OWLLiteral lit = (OWLLiteral) val;
                    if ((lit.hasLang() && lit.hasLang("en")) || !lit.hasLang()) {
                        String property_str = prop.getIRI().toString();
                        String e_str = e.getIRI().toString();
                        ann_axioms.add((e_str.replaceAll("\n", " ") + " " +
                                property_str.replaceAll("\n", " ") + " " +
                                ((OWLLiteral) val).getLiteral()).replaceAll("\n", " "));
                    }
                }
            }
        }
        save_annotations(ann_axioms);
    }

    private static String remove_datatype(String s) {
        ArrayList<String> datatypes = new ArrayList<>(Arrays.asList(
                "^^<http://www.w3.org/2001/XMLSchema#double>", "^^<http://www.w3.org/2001/XMLSchema#string>",
                "^^<http://www.w3.org/2001/XMLSchema#int>", "^^<http://www.w3.org/2001/XMLSchema#integer>",
                "^^<http://www.w3.org/2001/XMLSchema#float>"));
        for (String datatype : datatypes) {
            s = s.replace(datatype, "");
        }
        return s;
    }


    private static void save_axioms(ArrayList<String> sentences) throws IOException {
        FileWriter aw = new FileWriter(axiom_file);
        BufferedWriter abw = new BufferedWriter(aw);
        PrintWriter aout = new PrintWriter(abw);
        for (String s : sentences) {
            aout.println(s);
        }
        abw.close();
        aw.close();

    }

    private static void save_annotations(ArrayList<String> sentences) throws IOException {
        FileWriter aw = new FileWriter(annotation_file);
        BufferedWriter bw = new BufferedWriter(aw);
        PrintWriter out = new PrintWriter(bw);
        for (String s : sentences) {
            out.println(s);
        }
        bw.close();
        aw.close();
    }
}
