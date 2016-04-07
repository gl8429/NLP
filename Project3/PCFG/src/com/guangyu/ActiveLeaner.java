package com.guangyu;

import edu.stanford.nlp.parser.lexparser.EvaluateTreebank;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.parser.lexparser.Options;
import edu.stanford.nlp.trees.MemoryTreebank;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.Treebank;

import java.io.File;
import java.io.PrintStream;
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
//    private final static int[] SEED_SIZE = {20, 40};

    private final static int[] SELF_TRAINING = {1000, 2000, 3000, 4000, 5000, 7000, 10000, 13000, 17000, 21000};
//    private final static int[] SELF_TRAINING = {20, 40};

    private final static String[] WSJ_DIRS = {"02", "03", "04", "05", "06", "07", "08", "09", "10", "11"
            , "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22"};
//    private final static String[] WSJ_DIRS = {"02"};

    private final static String[] BROWN_DIRS = {"cf", "cg", "ck", "cl", "cm", "cn", "cp", "cr"};
//    private final static String[] BROWN_DIRS = {"cf"};

    private final static double RATE = 0.9;

    private final static int LABELED = 10000;
//    private final static int LABELED = 40;

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
                                              List<Tree> unlabeled, Treebank test, String name) {

        Map<String, String> result = new HashMap<>();

        Long time_started = System.currentTimeMillis();
        LexicalizedParser parser = LexicalizedParser.trainFromTreebank(wrapTreebank(initial), lpOptions);

        List<Tree> labeledByParse = new ArrayList<>();
        for (Tree tree : unlabeled) {
            labeledByParse.add(parser.apply(tree.yieldHasWord()));
        }
        initial.addAll(labeledByParse);

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

        try {
            System.setOut(new PrintStream(new File("trace/trees/" + name + ".txt")));
        } catch (Exception e) {
            e.printStackTrace();
        }

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

        PrintWriter writer = new PrintWriter("trace/result/output_wsj_no_wsj.txt");
        writer.println("\nNormal training and testing on WSJ.\n");
        wsj_no_wsj(writer);
        writer.close();

        writer = new PrintWriter("trace/result/output_wsj_no_brown.txt");
        writer.println("\nNormal training on WSJ and testing on Brown.\n");
        wsj_no_brown(writer);
        writer.close();

        writer = new PrintWriter("trace/result/output_wsj_brown_brown.txt");
        writer.println("\nUnsupervised domain adaption by normal training on WSJ, \nself-training on Brown, and " +
                "then testing on Brown\n");
        wsj_brown_brown(writer);
        writer.close();

        writer = new PrintWriter("trace/result/output_selfBrown.txt");
        writer.println("Self training of Brown");
        selfTrainingBrown(writer);
        writer.close();

        writer = new PrintWriter("trace/result/output_brown_no_brown.txt");
        writer.println("\nNormal training and testing on Brown.\n");
        brown_no_brown(writer);
        writer.close();

        writer = new PrintWriter("trace/result/output_brown_no_wsj.txt");
        writer.println("\nNormal training on Brown and testing on WSJ.\n");
        brown_no_wsj(writer);
        writer.close();

        writer = new PrintWriter("trace/result/output_brown_wsj_wsj");
        writer.println("\nUnsupervised domain adaption by normal training on Brown, \nself-training on WSJ, and " +
                "then testing on WSJ\n");
        brown_wsj_wsj(writer);
        writer.close();

        writer = new PrintWriter("trace/result/output_selfWSJ.txt");
        writer.println("Self training of WSJ");
        selfTrainingWSJ(writer);
        writer.close();
    }

    public static void wsj_no_wsj(PrintWriter writer){
        List<List<Tree>> initials = prepare_initial("/wsj/", WSJ_DIRS, SEED_SIZE);

        List<Tree> unlabeled = new ArrayList<>();

        MemoryTreebank test = new MemoryTreebank();
        test.loadPath(PATH + "wsj/23");
        test.textualSummary();

        printResults(computeEachSeed(initials, unlabeled, test, "WSJ_no_WSJ"), writer);
        System.out.println("Finished all seeds set.");
    }

    public static void brown_wsj_wsj(PrintWriter writer) {
        List<List<Tree>> data = separateBrown();
        List<Tree> seedSet = data.get(0);
        List<List<Tree>> initials = prepareFroBrown(seedSet);

        List<Tree> unlabeled = new ArrayList<>();

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

        printResults(computeEachSeed(initials, unlabeled, test, "Brown_WSJ_WSJ"), writer);
        System.out.println("Finished all seed set.");
    }

    public static void brown_no_brown(PrintWriter writer) {
        List<List<Tree>> data = separateBrown();
        List<Tree> seedSet = data.get(0);
        List<List<Tree>> initials = prepareFroBrown(seedSet);

        List<Tree> unlabeled = new ArrayList<>();

        List<Tree> testList = data.get(1);
        Treebank test = wrapTreebank(testList);

        printResults(computeEachSeed(initials, unlabeled, test, "Brown_no_Brown"), writer);
        System.out.println("Finished all seeds set");

    }

    public static void brown_no_wsj(PrintWriter writer) {
        List<List<Tree>> data = separateBrown();
        List<Tree> seedSet = data.get(0);
        List<List<Tree>> initials = prepareFroBrown(seedSet);

        List<Tree> unlabeled = new ArrayList<>();

        MemoryTreebank test = new MemoryTreebank();
        test.loadPath(PATH + "wsj/23");
        test.textualSummary();

        printResults(computeEachSeed(initials, unlabeled, test, "Brown_no_WSJ"), writer);
        System.out.println("Finished all seeds set");
    }

    public static List<List<Tree>> prepareFroBrown(List<Tree> dataSet) {
        List<List<Tree>> initials = new ArrayList<>();
        int index = 0;
        List<Tree> initial = new ArrayList<>();

        for (int seed : SELF_TRAINING) {
            while (initial.size() < seed)
                initial.add(dataSet.get(index++));
            initials.add(new ArrayList<>(initial));
        }
        return initials;
    }

    public static void selfTrainingBrown(PrintWriter writer) {
        List<Map<String, String>> results = new ArrayList<>();
        List<Tree> initial = prepare_initial("/wsj/", WSJ_DIRS, SEED_SIZE, LABELED);

        List<List<Tree>> brownDataSet = separateBrown();

        List<Tree> unlabeled = brownDataSet.get(0);
        List<Tree> test = brownDataSet.get(1);

        for (int i : SELF_TRAINING) {
            Map<String, String> result = iterate(initial, unlabeled.subList(0, i), wrapTreebank(test)
                    , "selfTrainingBrown");
            result.put("selfTraining", i + "");
            results.add(result);
        }
        printResults(results, writer);
        System.out.println("Finished all self training set.");
    }

    public static void selfTrainingWSJ(PrintWriter writer) {
        List<List<Tree>> data = separateBrown();
        List<Tree> initial = data.get(0).subList(0, LABELED);

        List<List<Tree>> unlabeled = prepare_initial("/wsj/", WSJ_DIRS, SEED_SIZE);

        MemoryTreebank test = new MemoryTreebank();
        test.loadPath(PATH + "wsj/23");
        test.textualSummary();

        printResults(computeEachSelfTraining(initial, unlabeled, test, "selfTrainingWSJ"), writer);
        System.out.println("Finished all self training set.");

    }

    public static List<Map<String, String>> computeEachSelfTraining(List<Tree> initial, List<List<Tree>> unlabeleds,
                                                                    Treebank test, String name) {
        List<Map<String, String>> results = new ArrayList<>();
        for (List<Tree> unlabeled : unlabeleds) {
            int size = initial.size();
            Map<String, String> result = iterate(initial, unlabeled, test, name);
            result.put("selfTraining", size + "");
            results.add(result);
        }
        return results;
    }

    public static List<List<Tree>> prepare_initial(String name, String[] dirs, int[] seeds) {
        List<List<Tree>> initials = new ArrayList<>();
        MemoryTreebank data = new MemoryTreebank();
        for (String dir : dirs) {
            String tmp = PATH + name + dir;
            data.loadPath(tmp);
        }
        data.textualSummary();
        int index = 0;
        List<Tree> initial = new ArrayList<>();
        for (int seed : seeds) {
            while (initial.size() < seed) {
                if (isTrainable(data.get(index))) initial.add(data.get(index));
                index++;
            }
            initials.add(new ArrayList<>(initial));
        }
        return initials;
    }

    public static List<Tree> prepare_initial(String name, String[] dirs, int[] seeds, int size) {
        List<List<Tree>> initials = prepare_initial(name, dirs, seeds);
        int index;
        for (index = 0; index < seeds.length; index++) {
            if (seeds[index] == size) break;
        }
        return initials.get(index);
    }

    public static void wsj_no_brown(PrintWriter writer){
        List<List<Tree>> initials = prepare_initial("/wsj/", WSJ_DIRS, SEED_SIZE);

        List<List<Tree>> brownDataSet = separateBrown();

        List<Tree> unlabeled = new ArrayList<>();
        List<Tree> testList = brownDataSet.get(1);

        Treebank test = wrapTreebank(testList);

        printResults(computeEachSeed(initials, unlabeled, test, "WSJ_no_Brown"), writer);
        System.out.println("Finished all seeds set");
    }

    public static void wsj_brown_brown(PrintWriter writer){
        List<List<Tree>> initials = prepare_initial("/wsj/", WSJ_DIRS, SEED_SIZE);

        List<List<Tree>> brownDataSet = separateBrown();

        List<Tree> unlabeled = brownDataSet.get(0);
        List<Tree> testList = brownDataSet.get(1);
        Treebank test = wrapTreebank(testList);

        printResults(computeEachSeed(initials, unlabeled, test, "WSJ_Brown_Brown"), writer);
        System.out.println("Finished all seeds set.");
    }

    private static List<Map<String, String>> computeEachSeed(List<List<Tree>> initials, List<Tree> unlabeled,
                                                             Treebank test, String name) {
        List<Map<String, String>> results = new ArrayList<>();

        for (List<Tree> current : initials) {
            int size = current.size();
            Map<String, String> result = iterate(current, unlabeled, test, name);
            result.put("seedSize", size + "");
            results.add(result);
        }
        return results;
    }

    private static List<List<Tree>> separateBrown() {
        List<Tree> unlabeled = new ArrayList<>();
        List<Tree> test = new ArrayList<>();

        for (String dir : BROWN_DIRS) {
            String cur = PATH + "brown/" + dir;
            List<List<Tree>> data = brownTrainTest(cur);
            unlabeled.addAll(data.get(0));
            test.addAll(data.get(1));
        }
        List<List<Tree>> result = new ArrayList<List<Tree>>(){};
        result.add(unlabeled);
        result.add(test);
        return result;
    }

    private static List<List<Tree>> brownTrainTest(String path) {
        MemoryTreebank brown = new MemoryTreebank();
        brown.loadPath(path);
        brown.textualSummary();
        List<Tree> dataSet = new ArrayList<>();
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