package com.guangyu;

import edu.stanford.nlp.parser.lexparser.EvaluateTreebank;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.parser.lexparser.Options;
import edu.stanford.nlp.trees.MemoryTreebank;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.Treebank;

import java.io.PrintWriter;
import java.util.*;

class TreebankWrapper {
    private List<Tree> trees;

    public TreebankWrapper(List<Tree> trees) {
        this.trees = trees;
    }

    public Treebank toTreebank() {
        Treebank treebank = new MemoryTreebank();
        treebank.addAll(trees);
        return treebank;
    }
}

public class ActiveLeaner {

    private final static Options lpOptions = new Options();

    private final static int[] SEED_SIZE = {1000, 2000, 3000, 4000, 5000, 7000, 10000, 13000, 16000, 20000, 25000, 30000, 35000};
//private final static int[] SEED_SIZE = {20, 40};

    private final static int[] SELF_TRAINING = {1000, 2000, 3000, 4000, 5000, 7000, 10000, 13000, 17000, 21000};
//    private final static int[] SELF_TRAINING = {20, 40};

    private final static String[] WSJ_DIRS = {"02", "03", "04", "05", "06", "07", "08", "09", "10", "11"
            , "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22"};
//    private final static String[] WSJ_DIRS = {"02"};

    private final static String[] BROWN_DIRS = {"cf", "cg", "ck", "cl", "cm", "cn", "cp", "cr"};
//    private final static String[] BROWN_DIRS = {"cf"};

    private final static double RATE = 0.9;

//    private final static int LABELED = 10000;
    private final static int LABELED = 40;

    private final static String PATH = "/Users/Lucifer/Documents/GraduateStudy/NLP/hw3_corpus/";

    public static void setLpOptions(Options lpOptions) {
        lpOptions.doDep = false;
        lpOptions.doPCFG = true;
        lpOptions.setOptions("-goodPCFG", "-evals", "tsv");
        lpOptions.testOptions.verbose = false;
    }

    public static Treebank wrapTreebank(List<Tree> trees) {
        TreebankWrapper treebankWrapper = new TreebankWrapper(trees);
        return treebankWrapper.toTreebank();
    }

    public static boolean isTrainable (Tree tree) {
        try {
            LexicalizedParser.trainFromTreebank(wrapTreebank(Collections.singletonList(tree)), lpOptions);
            return true;
        } catch (StringIndexOutOfBoundsException e) {
            return false;
        } catch (Exception e) {
            return false;
        }
    }

    public static Map<String, String> iterate(List<Tree> initial,
                                              List<Tree> unlabeled, Treebank test) {

        Map<String, String> result = new HashMap<String, String>();

        Long time_started = System.currentTimeMillis();
        LexicalizedParser parser = LexicalizedParser.trainFromTreebank(wrapTreebank(initial), lpOptions);

        initial.addAll(unlabeled);

        double active_training_words = 0;
        for (Tree tree : unlabeled) {
            active_training_words += tree.yieldHasWord().size();
        }
        double total_training_words = 0;
        for (Tree tree : initial) {
            total_training_words += tree.yieldHasWord().size();
        }

        LexicalizedParser retrained_parser;
        if (unlabeled.size() != 0) retrained_parser = LexicalizedParser.trainFromTreebank(wrapTreebank(initial), lpOptions);
        else retrained_parser = parser;

        result.put("added", active_training_words + "");
        result.put("total", total_training_words + "");
        // testOnTreebank
        EvaluateTreebank evaluateTreebank = new EvaluateTreebank(retrained_parser);
        result.put("f1", evaluateTreebank.testOnTreebank(test) + "");
        result.put("time", (System.currentTimeMillis() - time_started) / 1000 + "");

        return result;
    }

    public static void printResults(List<Map<String, String>> results, PrintWriter writer) {
        String seq = new String(new char[70]).replace("\0", "-");
        writer.println(seq);

        if (results.get(0).containsKey("selfTraining"))
            writer.format("%13s%12s%12s%12s%12s\n", "self-training", "added", "total", "f1", "time");
        if (results.get(0).containsKey("seedSize"))
            writer.format("%13s%12s%12s%12s%12s\n", "Seed Size", "added", "total", "f1", "time");

        for (Map<String, String> result : results) {
            if (result.containsKey("seedSize")) writer.format("%13s", result.get("seedSize"));
            if (result.containsKey("selfTraining")) writer.format("%13s", result.get("selfTraining"));
            writer.format("%12s", result.get("added"));
            writer.format("%12s", result.get("total"));
            writer.format("%12.3f", Double.parseDouble(result.get("f1")));
            writer.format("%12s\n", result.get("time"));
        }
        writer.println();

        writer.println(seq);
    }

    public static void main(String[] args) throws Exception{

        setLpOptions(lpOptions);

        PrintWriter writer = new PrintWriter("output_trace.txt");
        writer.println("\nNormal training and testing on WSJ.\n");
        wsj_no_wsj(writer);
        writer.println("\nNormal training on WSJ and testing on Brown.\n");
        wsj_no_brown(writer);
        writer.println("\nUnsupervised domain adaption by normal training on WSJ, \nself-training on Brown, and " +
                "then testing on Brown\n");
        wsj_brown_brown(writer);
        writer.println("Self training");
        self_training_diff(writer);
//        writer.println("\nUnsupervised domain adaption by normal training on Brown, \nself-training on WSJ, and " +
//                "then testing on WSJ\n");
//        brown_wsj_wsj(writer);

        writer.close();
    }

    public static void wsj_no_wsj(PrintWriter writer){
        List<Map<String, String>> results = new ArrayList<Map<String, String>>();

        List<List<Tree>> initials = prepareWSJ_initial();

        List<Tree> unlabeled = new ArrayList<Tree>();

        MemoryTreebank test = new MemoryTreebank();
        test.loadPath(PATH + "wsj/23");
        test.textualSummary();
        for (List<Tree> current : initials) {
            Map<String, String> result = iterate(current, unlabeled, test);
            result.put("seedSize", current.size() + "");
            results.add(result);
        }
        printResults(results, writer);
        System.out.println("Finished all seeds set.");
    }

    public static List<List<Tree>> prepareWSJ_initial() {
        List<List<Tree>> initials = new ArrayList<List<Tree>>();
        MemoryTreebank wsj = new MemoryTreebank();
        for (String dir : WSJ_DIRS) {
            String tmp = PATH + "wsj/" + dir;
            wsj.loadPath(tmp);
        }
        wsj.textualSummary();
        int index = 0;
        List<Tree> initial = new ArrayList<Tree>();
        for (int seed : SEED_SIZE) {
            while (initial.size() < seed) {
                if (isTrainable(wsj.get(index))) initial.add(wsj.get(index));
                index++;
            }
            initials.add(new ArrayList<Tree>(initial));
        }
        return initials;
    }

    public static List<Tree> prepareWSJ_initial(int size) {
        List<List<Tree>> initials = prepareWSJ_initial();
        int index;
        for (index = 0; index < SEED_SIZE.length; index++) {
            if (SEED_SIZE[index] == size) break;
        }
        return initials.get(index);
    }

    public static void brown_wsj_wsj(PrintWriter writer) {
        List<Map<String, String>> results = new ArrayList<Map<String, String>>();
        List<List<Tree>> initials = prepareBrown_initial();

        List<Tree> unlabeled = new ArrayList<Tree>();

        MemoryTreebank wsj = new MemoryTreebank();
        for (String dir : WSJ_DIRS) {
            String tmp = PATH + "wsj/" + dir;
            wsj.loadPath(tmp);
        }
        wsj.textualSummary();
        for (Tree tree : wsj) {
            if (isTrainable(tree)) unlabeled.add(tree);
        }

        MemoryTreebank test = new MemoryTreebank();
        test.loadPath(PATH + "wsj/23");
        test.textualSummary();
        for (List<Tree> initial : initials) {
            Map<String, String> result = iterate(initial, unlabeled, test);
            result.put("seedSize", initial.size() + "");
            results.add(result);
        }
        printResults(results, writer);
        System.out.println("Finished all seed set.");
    }

    public static void brown_no_brown(PrintWriter writer) {

    }

    public static List<List<Tree>> prepareBrown_initial() {
        List<List<Tree>> initials = new ArrayList<List<Tree>>();

        MemoryTreebank brown = new MemoryTreebank();
        for (String dir : BROWN_DIRS) {
            String cur = PATH + "brown/" + dir;
            brown.loadPath(cur);
        }
        brown.textualSummary();
        int index = 0;
        List<Tree> initial = new ArrayList<Tree>();
        for (int seed : SELF_TRAINING) {
            while (initial.size() < seed) {
                if (isTrainable(brown.get(index))) initial.add(brown.get(index));
                index++;
            }
            initials.add(initial.subList(0, seed));
        }
        return initials;
    }


    public static List<Tree> prepareBrown_initial(int size) {
        List<List<Tree>> initials = prepareBrown_initial();
        int index;
        for (index = 0; index < SELF_TRAINING.length; index++) {
            if (SELF_TRAINING[index] == size) break;
        }
        return initials.get(index);
    }

    public static void self_training_diff(PrintWriter writer) {
        List<Map<String, String>> results = new ArrayList<Map<String, String>>();
        List<Tree> initial = prepareWSJ_initial(LABELED);

        List<Tree> unlabeled = new ArrayList<Tree>();
        List<Tree> test = new ArrayList<Tree>();

        for (String dir : BROWN_DIRS) {
            String cur = PATH + "brown/" + dir;
            List<List<Tree>> data = brownTrainTest(cur);
            unlabeled.addAll(data.get(0));
            test.addAll(data.get(1));
        }

        for (int i : SELF_TRAINING) {
            Map<String, String> result = iterate(initial, unlabeled.subList(0, i), wrapTreebank(test));
            result.put("selfTraining", i + "");
            results.add(result);
        }
        printResults(results, writer);
        System.out.println("Finished all self training set.");
    }

    public static void wsj_no_brown(PrintWriter writer){
        List<Map<String, String>> results = new ArrayList<Map<String, String>>();
        List<List<Tree>> initials = prepareWSJ_initial();

        List<Tree> unlabeled = new ArrayList<Tree>();
        List<Tree> test = new ArrayList<Tree>();

        for (String dir : BROWN_DIRS) {
            String cur = PATH + "brown/" + dir;
            List<List<Tree>> data = brownTrainTest(cur);
            test.addAll(data.get(1));
        }

        for (List<Tree> current : initials) {
            int size = current.size();
            Map<String, String> result = iterate(current, unlabeled, wrapTreebank(test));
            result.put("seedSize", size + "");
            results.add(result);
        }
        printResults(results, writer);
        System.out.println("Finished all seeds set");
    }

    public static void wsj_brown_brown(PrintWriter writer){
        List<Map<String, String>> results = new ArrayList<Map<String, String>>();
        List<List<Tree>> initials = prepareWSJ_initial();

        List<Tree> unlabeled = new ArrayList<Tree>();
        List<Tree> test = new ArrayList<Tree>();

        for (String dir : BROWN_DIRS) {
            String cur = PATH + "brown/" + dir;
            List<List<Tree>> data = brownTrainTest(cur);
            unlabeled.addAll(data.get(0));
            test.addAll(data.get(1));
        }

        for (List<Tree> current : initials) {
            int size = current.size();
            Map<String, String> result = iterate(current, unlabeled, wrapTreebank(test));
            result.put("seedSize", size + "");
            results.add(result);
        }
        printResults(results, writer);
        System.out.println("Finished all seeds set.");
    }

    public static List<List<Tree>> brownTrainTest(String path) {
        MemoryTreebank brown = new MemoryTreebank();
        brown.loadPath(path);
        brown.textualSummary();
        List<Tree> dataSet = new ArrayList<Tree>();
        for (Tree data : brown) {
            if (isTrainable(data)) dataSet.add(data);
        }
        int slice = (int) (dataSet.size() * RATE);
        List<Tree> train = dataSet.subList(0, slice);
        List<Tree> test = dataSet.subList(slice, dataSet.size());
        List<List<Tree>> result = new ArrayList<List<Tree>>(){};
        result.add(train);
        result.add(test);
        return result;
    }
}