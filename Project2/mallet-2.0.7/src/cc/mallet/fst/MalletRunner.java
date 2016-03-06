package cc.mallet.fst;

//import cc.mallet.fst.HMMSimpleTagger;
//import cc.mallet.fst.SimpleTagger;
import cc.mallet.pipe.Pipe;
import cc.mallet.types.*;

import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

class OOVTokenAccuracyEvaluator extends TokenAccuracyEvaluator {

	public OOVTokenAccuracyEvaluator (InstanceList[] instanceLists, String[] descriptions) {
		super (instanceLists, descriptions);
	}

    public String[] tokensForInstance(Instance instance) {

        FeatureSequence featureSequence = (FeatureSequence)instance.getData();
        String[] list = new String[featureSequence.size()];
        for (int i = 0; i < featureSequence.size(); ++i) {
            list[i] = featureSequence.get(i).toString();
        }
        return list;
    }

	public Double[] oovEvaluateInstanceList(Transducer transducer, InstanceList instances, Set<String> vocabulary) {

        int numCorrectOOV;
        int totalOOV;
        int totalTokens;
        int numCorrectTokens;

        numCorrectOOV = totalOOV = totalTokens = numCorrectTokens = 0;
        for (int i = 0; i < instances.size(); ++i) {
            Instance instance = instances.get(i);
            Sequence input = (Sequence) instance.getData();
            String[] tokens = tokensForInstance(instance);
            Sequence gold_labels = (Sequence) instance.getTarget();
            assert(input.size() == gold_labels.size());

            Sequence output_labels = transducer.transduce(input);
            assert(gold_labels.size() == output_labels.size());

            List<String[]> seq = new ArrayList<String[]>();
            for (int j = 0; j < input.size(); ++j) {
                seq.add(new String[]{tokens[j], (String)gold_labels.get(j), (String) output_labels.get(j)});
            }

            List<String[]> oov_seq = new ArrayList<String[]>();
            for (String[] s : seq) {
                totalTokens++;
                if (!vocabulary.contains(s[0])) {
                    totalOOV++;
                    oov_seq.add(s);
                    if (s[2].equals(s[1])) numCorrectOOV++;
                }
                if (s[2].equals(s[1])) numCorrectTokens++;
            }
        }
        double acc = ((double) numCorrectTokens) / totalTokens;
        double ovv_acc = ((double) numCorrectOOV) / totalOOV;
        return new Double[]{acc, ovv_acc};
    }
}

class PREOOVTokenAccuracyEvaluator extends OOVTokenAccuracyEvaluator {

    public PREOOVTokenAccuracyEvaluator(InstanceList[] instanceLists, String[] descriptions) {
        super(instanceLists, descriptions);
    }

    @Override
    public String[] tokensForInstance(Instance instance) {
        FeatureVectorSequence featureSequence = (FeatureVectorSequence)instance.getData();
        String[] list = new String[featureSequence.size()];
        for (int i = 0; i < featureSequence.size(); ++i) {
            FeatureVector fv = featureSequence.get(i);
            list[i] = fv.getAlphabet().lookupObject(fv.indexAtLocation(0)).toString();
        }
        return list;
    }
}

public class MalletRunner {

    public static List<String> reverse(List<String> sentences) {
        List<String> res = new ArrayList<String>();
        for (String sentence : sentences) {
            List<String> tmp = Arrays.asList(sentence.split("\n"));
            Collections.reverse(tmp);
            res.add(String.join("\n", tmp));
        }
        return res;
    }

    public static List<String> addFeatures(List<String> sentences) {
        List<String> res = new ArrayList<String>();
        for (String sentence : sentences) {
            List<String> tmp = new ArrayList<String>();
            for (String words : sentence.split("\n")) {
                tmp.add(addFeature(words));
            }
            res.add(String.join("\n", tmp));
        }
        return res;
    }

    public static String addFeature(String words) {
        String[] p = words.split(" ");
        String token = p[0];
        List<String> res = tagToken(token);
        res.add(0, token);
        res.add(p[1]);
        return String.join(" ", res);
    }

    public static List<String> tagToken(String token) {
        List<String> res = new ArrayList<String>();
        if (token.length() < 4) res.add("SHORT");
        if (token.charAt(0) >= 'A' && token.charAt(0) <= 'Z') res.add("CAPS");
        return res;
    }

    public static List<List<String>> getTrainTestSentences(String trainDirectory, String testDirectory, Double trainProportion
            , Boolean extraFeatures, Boolean reverse, Long randomSeed)throws Exception{

        List<String> sentences_train = null;
        List<String> sentences_test = null;
        if (trainProportion != null) {
            String trainingFile = new String(Files.readAllBytes(Paths.get(trainDirectory)));
            List<String> sentences = Arrays.asList(trainingFile.split("\n{2,}"));
            int numSentences = sentences.size();
            int trainingCount = (int)Math.round(trainProportion * numSentences);
            if (randomSeed != null) Collections.shuffle(sentences, new Random(randomSeed));
            sentences_train = sentences.subList(0, trainingCount);
            sentences_test = sentences.subList(trainingCount, numSentences);
        } else if (testDirectory != null){
            String trainingFile = new String(Files.readAllBytes(Paths.get(trainDirectory)));
            sentences_train = Arrays.asList(trainingFile.split("\n{2,}"));
            String testFile = new String(Files.readAllBytes(Paths.get(testDirectory)));
            sentences_test = Arrays.asList(testFile.split("\n{2,}"));
        }

        if (reverse) {
            sentences_train = reverse(sentences_train);
            sentences_test = reverse(sentences_test);
        }
        if (extraFeatures) {
            sentences_train = addFeatures(sentences_train);
            sentences_test = addFeatures(sentences_test);
        }

        List<List<String>> res = new ArrayList<List<String>>();
        res.add(sentences_train);
        res.add(sentences_test);
        return res;
    }

    public static InstanceList convertToInstance(List<String> sentences, Pipe pipe) {
        InstanceList instances = new InstanceList(pipe);
        for (String sentence : sentences) {
            instances.add(pipe.pipe(new Instance(sentence, null, null, null)));
        }
        return instances;
    }

    public static String[] toStringArray(Object[] objects) {
        String[] stringArray = new String[objects.length];
        for (int i = 0; i < objects.length; ++i) {
            stringArray[i] = (String)objects[i];
        }
        return stringArray;
    }

	public static void main(String[] args) throws Exception{
		String model = args[0].toLowerCase();
		String train_dir = args[1];
		String test_dir = args[2];
		Double train_rate = Double.parseDouble(args[3]);
        Boolean extra = (args[4].equals("1"));
        Boolean reverse = args[5].equals("1");
		int folds = Integer.parseInt(args[6]);
        if (test_dir.equals("none")) test_dir = null;
        else train_rate = null;

        int iterations = 500;
        
        PrintWriter writter = new PrintWriter(train_dir.substring(train_dir.lastIndexOf('/') + 1) + "-" + model
                + "-extra" + extra + "-iter" + iterations + ".txt");

        double time = 0, train = 0, test = 0, oov = 0;

        for (int i = 0; i < folds; ++i) {
            double timeStarted = System.currentTimeMillis();
            Pipe simplePipe;
            if (model.equals("crf")) simplePipe = new SimpleTagger.SimpleTaggerSentence2FeatureVectorSequence();
            else simplePipe = new HMMSimpleTagger.HMMSimpleTaggerSentence2FeatureSequence();
            simplePipe.getTargetAlphabet().lookupIndex("0");
            List<List<String>> allSentences = getTrainTestSentences(train_dir, test_dir, train_rate, extra,
                    reverse, System.nanoTime());
            InstanceList trainInstances = convertToInstance(allSentences.get(0), simplePipe);
            Set<String> trainVocabulary = new HashSet<String>();
            for (String vocabulary : toStringArray(simplePipe.getDataAlphabet().toArray())) trainVocabulary.add(vocabulary);
            InstanceList testInstances = convertToInstance(allSentences.get(1), simplePipe);
            OOVTokenAccuracyEvaluator evaluator;
            Transducer transducer;
            if (model.equals("crf")) {
                evaluator = new PREOOVTokenAccuracyEvaluator(new InstanceList[]{trainInstances, testInstances}
                , new String[]{"Training", "Testing"});
                transducer = SimpleTagger.train(trainInstances, testInstances, null, new int[]{1}, "0", "\\s", ".*",
                        true, iterations, 10.0, null);
            } else {
                evaluator = new OOVTokenAccuracyEvaluator(new InstanceList[]{trainInstances, testInstances}
                        , new String[]{"Training", "Testing"});
                transducer = HMMSimpleTagger.train(trainInstances, testInstances, null, new int[]{1}, "0", "\\s", ".*",
                        true, iterations, 10.0, null);
            }
            double timeTrained = System.currentTimeMillis();
            Double[] trainingAcc = evaluator.oovEvaluateInstanceList(transducer, trainInstances, new HashSet<String>());
            Double[] testAcc = evaluator.oovEvaluateInstanceList(transducer, testInstances, trainVocabulary);

            double timeEnded = System.currentTimeMillis();

            writter.println("\n\n============== " + i + " =============\n");
            writter.println("Training Time: " + (timeTrained - timeStarted) / 1000.0 + "\n"
                    + "Testing Time: " + (timeEnded - timeTrained) / 1000.0 + "\n" + "Total Time: "
                    + (timeEnded - timeStarted) / 1000.0);
            time += (timeEnded - timeStarted) / 1000.0;
            writter.println("Training sentence: " + trainInstances.size());
            writter.println("Test sentence: " + testInstances.size());
            writter.println("Training Accuracy: " + trainingAcc[0]);
            train += trainingAcc[0];
            writter.println("Test Accuracy: " + testAcc[0]);
            test += testAcc[0];
            writter.println("OOV Accuracy: " + testAcc[1]);
            oov += testAcc[1];
        }
        writter.println("\nAverage time: " + time / folds + "\nTraining Accuracy: " + train / folds
                + "\nTest Accuracy: " + test / folds + "\noov: " + oov / folds);
        writter.close();
    }
}