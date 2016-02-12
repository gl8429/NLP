package nlp.lm;

import java.io.*;
import java.util.*;

/**
 *
 *
 * @author Guangyu Lin
 * Combines the forward and backward model to build a n-grams language model.
 * Interpolates both models with equal weights
 */

public class BidirectionalBigramModel {

    /** Forward bigram model */
    private BigramModel bigramModel;

    /** Backward bigram model */
    private BackwardBigramModel backwardBigramModel;

    /** Initialize Bidirectional Bigram Model */
    public BidirectionalBigramModel() {
        bigramModel = new BigramModel();
        backwardBigramModel = new BackwardBigramModel();
    }

    /** Train Bidirectional Bigram Model by training sub models. */
    public void train(List<List<String>> sentences) {
        bigramModel.train(sentences);
        backwardBigramModel.train(sentences);
    }

    /** Use sentences as a test set to evaluate the model. Print out perplexity
     *  of the model for this test data */
    public void test (List<List<String>> sentences) {
        // Compute log probability of sentence to avoid underflow
        double totalLogProb = 0;
        // Keep count of total number of tokens predicted
        double totalNumTokens = 0;
        // Accumulate log prob of all test sentences
        for (List<String> sentence : sentences) {
            // Num of tokens in sentence plus 1 for predicting </S>
            totalNumTokens += sentence.size() + 1;
            // Compute log prob of sentence
            double sentenceLogProb = sentenceLogProb(sentence);
            //      System.out.println(sentenceLogProb + " : " + sentence);
            // Add to total log prob (since add logs to multiply probs)
            totalLogProb += sentenceLogProb;
        }
        // Given log prob compute perplexity
        double perplexity = Math.exp(-totalLogProb / totalNumTokens);
        System.out.println("Perplexity = " + perplexity );
    }

    /** Compute log probability of sentence given current model */
    public double sentenceLogProb (List<String> sentence) {

    	/** Compute probabilities vector for both sub models. */
    	double[] forwardProbs = bigramModel.sentenceTokenProbs(sentence);
    	double[] backwardProbs = backwardBigramModel.sentenceTokenProbs(sentence);

	    /** Maintain total sentence prob as sum of individual token
	        log probs (since adding logs is same as multiplying probs) */
	    double sentenceLogProb = 0;
	    /** Check prediction of each token in sentence */
	    for (int i = 0; i < sentence.size(); ++i) {
	    	/** Probabilities of ith token in the sentence of forward model */
	    	double forwardProbToken = forwardProbs[i];

	    	/** Probabilities of ith token in the sentence of backward model, except
	    		the sentence's start probabilities at the end */
	    	double backwardProbToken = backwardProbs[backwardProbs.length - i - 2];

 			/** Compute log prob of token using forward and backward model (equal weight) */
	    	double logProb = Math.log((forwardProbToken + backwardProbToken) / 2);

	    	/** Add token log prob to sentence log prob */
	        sentenceLogProb += logProb;
	    }
	    /** Calculate sentence boundary prediction */
	    double endSentenceProb = forwardProbs[sentence.size()];
	    double startSentenceProb = backwardProbs[sentence.size()];
	    sentenceLogProb += Math.log((endSentenceProb + startSentenceProb) / 2);
	    return sentenceLogProb;
    }

    /** Like test1 but excludes predicting end-of-sentence when computing perplexity */
    public void test2 (List<List<String>> sentences) {
	    double totalLogProb = 0;
	    double totalNumTokens = 0;
	    for (List<String> sentence : sentences) {
	        totalNumTokens += sentence.size();
	        double sentenceLogProb = sentenceLogProb2(sentence);
	        //      System.out.println(sentenceLogProb + " : " + sentence);
	        totalLogProb += sentenceLogProb;
	    }
	    double perplexity = Math.exp(-totalLogProb / totalNumTokens);
	    System.out.println("Word Perplexity = " + perplexity );
    }

    /** Like sentenceLogProb but excludes predicting end-of-sentence when computing prob */
    public double sentenceLogProb2 (List<String> sentence) {

    	double[] forwardProbs = bigramModel.sentenceTokenProbs(sentence);
    	double[] backwardProbs = backwardBigramModel.sentenceTokenProbs(sentence);

	    double sentenceLogProb = 0;

	    for (int i = 0; i < sentence.size(); ++i) {
	    	double forwardProbToken = forwardProbs[i];
	    	double backwardProbToken = backwardProbs[backwardProbs.length - i - 2];
	    	double logProb = Math.log((forwardProbToken + backwardProbToken) / 2);
	        sentenceLogProb += logProb;
	    }
	    return sentenceLogProb;
    }

        /** Train and test a bigram model.
     *  Command format: "nlp.lm.BigramModel [DIR]* [TestFrac]" where DIR 
     *  is the name of a file or directory whose LDC POS Tagged files should be 
     *  used for input data; and TestFrac is the fraction of the sentences
     *  in this data that should be used for testing, the rest for training.
     *  0 < TestFrac < 1
     *  Uses the last fraction of the data for testing and the first part
     *  for training.
     */
    public static void main(String[] args) throws IOException {
	    // All but last arg is a file/directory of LDC tagged input data
	    File[] files = new File[args.length - 1];
	    for (int i = 0; i < files.length; i++) 
	        files[i] = new File(args[i]);
	    // Last arg is the TestFrac
	    double testFraction = Double.valueOf(args[args.length -1]);
	    // Get list of sentences from the LDC POS tagged input files
	    List<List<String>> sentences =  POSTaggedFile.convertToTokenLists(files);
	    int numSentences = sentences.size();
	    // Compute number of test sentences based on TestFrac
	    int numTest = (int)Math.round(numSentences * testFraction);
	    // Take test sentences from end of data
	    List<List<String>> testSentences = sentences.subList(numSentences - numTest, numSentences);
	    // Take training sentences from start of data
	    List<List<String>> trainSentences = sentences.subList(0, numSentences - numTest);
	    System.out.println("# Train Sentences = " + trainSentences.size() + 
	               " (# words = " + BigramModel.wordCount(trainSentences) + 
	               ") \n# Test Sentences = " + testSentences.size() +
	               " (# words = " + BigramModel.wordCount(testSentences) + ")");
	    // Create a bigram model and train it.
	    BidirectionalBigramModel model = new BidirectionalBigramModel();
	    System.out.println("Training...");
	    model.train(trainSentences);
	    // Test on training data using test and test2
	    model.test(trainSentences);
	    model.test2(trainSentences);
	    System.out.println("Testing...");
	    // Test on test data using test and test2
	    model.test(testSentences);
	    model.test2(testSentences);
    }
}
	
