package cc.mallet.fst;

//import cc.mallet.fst.HMMSimpleTagger;
//import cc.mallet.fst.SimpleTagger;
import cc.mallet.pipe.Pipe;
import cc.mallet.types.*;

import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

/**
 *  @author Guangyu Lin
 *  An integrated model that can run HMM and CRF with input param
 *  , which supports adding features, forwarding and backwarding
 *  sencentances, running times, training rate.
 *  Output record includes, overall training accuracy, test accuracy,
 *  out-of-vocabulary accuracy and running time.
 */

/** Extends TokenAccuracyEvaluator to calculate out-of-vocabulary accuracy
 *  , model: HMM
 * */

class OOVTokenAccuracyEvaluator extends TokenAccuracyEvaluator {

    /** Initialize OOVTokenAccuracyEvaluator */

	public OOVTokenAccuracyEvaluator (InstanceList[] instanceLists, String[] descriptions) {
		super (instanceLists, descriptions);
	}

    /** Get all input tokens */

    public String[] tokensForInstance(Instance instance) {

        FeatureSequence featureSequence = (FeatureSequence)instance.getData();
        String[] list = new String[featureSequence.size()];
        for (int i = 0; i < featureSequence.size(); ++i) {
            list[i] = featureSequence.get(i).toString();
        }
        return list;
    }

    /**
     *  Calculate overall accuracy of training and test and out-of-vocabulary accuracy
     *  @reutrn [overall accuracy, oov accuracy]
     * */

	public Double[] oovEvaluateInstanceList(Transducer transducer, InstanceList instances, Set<String> vocabulary) {

        int numCorrectOOV;
        int totalOOV;
        int totalTokens;
        int numCorrectTokens;

        numCorrectOOV = totalOOV = totalTokens = numCorrectTokens = 0;
        for (int i = 0; i < instances.size(); ++i) {
            Instance instance = instances.get(i);
            Sequence input = (Sequence) instance.getData();
            // get input tokens
            String[] tokens = tokensForInstance(instance);
            // target output
            Sequence gold_labels = (Sequence) instance.getTarget();
            assert(input.size() == gold_labels.size());
            // predictions
            Sequence output_labels = transducer.transduce(input);
            assert(gold_labels.size() == output_labels.size());

            List<String[]> seq = new ArrayList<String[]>();
            for (int j = 0; j < input.size(); ++j) {
                seq.add(new String[]{tokens[j], (String)gold_labels.get(j), (String) output_labels.get(j)});
            }

            for (String[] s : seq) {
                totalTokens++;
                // if it is a out-of-vocabulary token
                if (!vocabulary.contains(s[0])) {
                    totalOOV++;
                    // if prediction is correct
                    if (s[2].equals(s[1])) numCorrectOOV++;
                }
                // correct prediction's record for overall
                if (s[2].equals(s[1])) numCorrectTokens++;
            }
        }
        double acc = ((double) numCorrectTokens) / totalTokens;
        double ovv_acc = ((double) numCorrectOOV) / totalOOV;
        return new Double[]{acc, ovv_acc};
    }
}

/** Extends TokenAccuracyEvaluator to calculate out-of-vocabulary accuracy
 *  , model: CRF
 * */

class PREOOVTokenAccuracyEvaluator extends OOVTokenAccuracyEvaluator {

    /** Initialize PREOOVTokenAccuracyEvaluator */

    public PREOOVTokenAccuracyEvaluator(InstanceList[] instanceLists, String[] descriptions) {
        super(instanceLists, descriptions);
    }

    /** Get all input tokens, which is an override function caused by different types of two models */

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

/** An integrated application */

public class MalletRunner {

    /** Reverse the sentence to implement backward */

    private static List<String> reverse(List<String> sentences) {
        List<String> res = new ArrayList<String>();
        for (String sentence : sentences) {
            // words in sentence is separated by `\n`
            List<String> tmp = Arrays.asList(sentence.split("\n"));
            Collections.reverse(tmp);
            res.add(String.join("\n", tmp));
        }
        return res;
    }

    /** Add features for each sentence */

    private static List<String> addFeatures(List<String> sentences) {
        List<String> res = new ArrayList<String>();
        for (String sentence : sentences) {
            List<String> tmp = new ArrayList<String>();
            // words in sentence is separated by `\n`
            for (String words : sentence.split("\n")) {
                tmp.add(addFeature(words));
            }
            res.add(String.join("\n", tmp));
        }
        return res;
    }

    /** Split tokenize words into word and tokenize each word by more features */

    private static String addFeature(String words) {
        String[] p = words.split(" ");
        String token = p[0];
        List<String> res = tagToken(token);
        res.add(0, token);
        res.add(p[1]);
        // pack words with new features
        return String.join(" ", res);
    }

    /** Add extra features to tokens
     *  @return list of matched features
     * */

    private static List<String> tagToken(String token) {
        List<String> res = new ArrayList<String>();
        int index;
        // if word's length is shorter than 4
        if (token.length() < 4) res.add("SHORT");
        // if first letter is capital
        if (token.charAt(0) >= 'A' && token.charAt(0) <= 'Z') res.add("CAPS");
        index = token.length() - 3 >= 0 ? token.length() - 3 : 0;
        // if progressive tense
        if (token.substring(index).equals("ing")) res.add("ING");
        // if plural
        if (token.substring(token.length() - 1).equals("s")) res.add("S");
        index = token.length() - 2 >= 0 ? token.length() - 2 : 0;
        // if past tense
        if (token.substring(index).equals("ed")) res.add("PAST");
        return res;
    }

    /** Retrieve sentences for training and test.
     *
     *  @input training director, test directory, training proportion, whether adding extra features
     *         , forward or backward, random seed
     *  @return [training sentences, test sentences]
     *  If there is a test data directory, retrieve all sentences in test and train
     *  otherwise, calculate training sentences and test sentences by the given proportion and shuffle.
     *
     * */

    public static List<List<String>> getTrainTestSentences(String trainDirectory, String testDirectory, Double trainProportion
            , Boolean extraFeatures, Boolean reverse, Long randomSeed)throws Exception{

        List<String> sentences_train = null;
        List<String> sentences_test = null;
        // if there is a train proportion
        if (trainProportion != null) {
            String trainingFile = new String(Files.readAllBytes(Paths.get(trainDirectory)));
            // retrieve all sentences in file, each sentences are split by two `\n`s
            List<String> sentences = Arrays.asList(trainingFile.split("\n{2,}"));
            int numSentences = sentences.size();
            // calculate the training size
            int trainingCount = (int)Math.round(trainProportion * numSentences);
            // shuffle sentences by time seed
            if (randomSeed != null) Collections.shuffle(sentences, new Random(randomSeed));
            sentences_train = sentences.subList(0, trainingCount);
            sentences_test = sentences.subList(trainingCount, numSentences);
        // if there is a test directory
        } else if (testDirectory != null){
            String trainingFile = new String(Files.readAllBytes(Paths.get(trainDirectory)));
            // retrieve all sentences in file, each sentences are split by two `\n`s
            sentences_train = Arrays.asList(trainingFile.split("\n{2,}"));
            String testFile = new String(Files.readAllBytes(Paths.get(testDirectory)));
            sentences_test = Arrays.asList(testFile.split("\n{2,}"));
        }
        // if a backward model
        if (reverse) {
            sentences_train = reverse(sentences_train);
            sentences_test = reverse(sentences_test);
        }
        // if more features are needed
        if (extraFeatures) {
            sentences_train = addFeatures(sentences_train);
            sentences_test = addFeatures(sentences_test);
        }
        // return a training sentences and test sentences
        List<List<String>> res = new ArrayList<List<String>>();
        res.add(sentences_train);
        res.add(sentences_test);
        return res;
    }

    /** Convert sentences to instances in order to prepare for training and testing */

    public static InstanceList convertToInstance(List<String> sentences, Pipe pipe) {
        InstanceList instances = new InstanceList(pipe);
        for (String sentence : sentences) {
            instances.add(pipe.pipe(new Instance(sentence, null, null, null)));
        }
        return instances;
    }

    /** Covert an object array into String array */

    private static String[] toStringArray(Object[] objects) {
        String[] stringArray = new String[objects.length];
        for (int i = 0; i < objects.length; ++i) {
            stringArray[i] = (String)objects[i];
        }
        return stringArray;
    }

    /** Input arguments should clearly define as following,
     *  1. models: HMM or CRF
     *  2. training directory
     *  3. training proportion,
     *  4. test directory,
     *  5. extra features: 1(Yes) or 0(No)
     *  6. forward model: 1(Yes) or 0(No)
     *  7. folds: repeat times
      */

	public static void main(String[] args) throws Exception{
        // store input arguments
		String model = args[0].toLowerCase();
		String train_dir = args[1];
		String test_dir = args[2];
		Double train_rate = Double.parseDouble(args[3]);
        Boolean extra = (args[4].equals("1"));
        Boolean reverse = args[5].equals("1");
		int folds = Integer.parseInt(args[6]);
        if (test_dir.equals("none")) test_dir = null;
        else train_rate = null;

        // initial iterations is set as 500
        int iterations = 500;

        // write result into a file and give an identity name
        PrintWriter writter = new PrintWriter(train_dir.substring(train_dir.lastIndexOf('/') + 1) + "-" + model
                + "-extra" + extra + "-iter" + iterations + "-re" + reverse +".txt");

        // calculate the average when more than 1 run
        double time = 0, train = 0, test = 0, oov = 0;

        for (int i = 0; i < folds; ++i) {
            // record the start time
            double timeStarted = System.currentTimeMillis();
            Pipe simplePipe;
            // running CRF
            if (model.equals("crf")) simplePipe = new SimpleTagger.SimpleTaggerSentence2FeatureVectorSequence();
            // running HMM
            else simplePipe = new HMMSimpleTagger.HMMSimpleTaggerSentence2FeatureSequence();
            // tricky part
            simplePipe.getTargetAlphabet().lookupIndex("0");
            // get training senteces and test sentences
            List<List<String>> allSentences = getTrainTestSentences(train_dir, test_dir, train_rate, extra,
                    reverse, System.nanoTime());
            // prepare sentences as instances for training
            InstanceList trainInstances = convertToInstance(allSentences.get(0), simplePipe);
            // store all training tokens for calculating oov accuracy later
            Set<String> trainVocabulary = new HashSet<String>();
            // remove duplicate tokens
            for (String vocabulary : toStringArray(simplePipe.getDataAlphabet().toArray())) trainVocabulary.add(vocabulary);
            // prepare sentences as instances for testin
            InstanceList testInstances = convertToInstance(allSentences.get(1), simplePipe);
            OOVTokenAccuracyEvaluator evaluator;
            Transducer transducer;
            if (model.equals("crf")) {
                // initialize evaluator and transducer for CRF
                evaluator = new PREOOVTokenAccuracyEvaluator(new InstanceList[]{trainInstances, testInstances}
                , new String[]{"Training", "Testing"});
                transducer = SimpleTagger.train(trainInstances, testInstances, null, new int[]{1}, "0", "\\s", ".*",
                        true, iterations, 10.0, null);
            } else {
                // initialize evaluator and transducer for HMM
                evaluator = new OOVTokenAccuracyEvaluator(new InstanceList[]{trainInstances, testInstances}
                        , new String[]{"Training", "Testing"});
                transducer = HMMSimpleTagger.train(trainInstances, testInstances, null, new int[]{1}, "0", "\\s", ".*",
                        true, iterations, 10.0, null);
            }
            // time checkpoint to calculate training time
            double timeTrained = System.currentTimeMillis();
            // calculate the overall accuracy of training and test and specific oov accuracy
            Double[] trainingAcc = evaluator.oovEvaluateInstanceList(transducer, trainInstances, new HashSet<String>());
            Double[] testAcc = evaluator.oovEvaluateInstanceList(transducer, testInstances, trainVocabulary);

            // time checkpoint to get total time
            double timeEnded = System.currentTimeMillis();

            // output format
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
        // output average data
        writter.println("\nAverage time: " + time / folds + "\nTraining Accuracy: " + train / folds
                + "\nTest Accuracy: " + test / folds + "\noov: " + oov / folds);
        writter.close();
    }
}