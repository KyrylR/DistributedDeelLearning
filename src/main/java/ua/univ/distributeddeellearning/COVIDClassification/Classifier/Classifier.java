package ua.univ.distributeddeellearning.COVIDClassification.Classifier;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ua.univ.distributeddeellearning.COVIDClassification.Configuration.ConfigDLParameters;
import ua.univ.distributeddeellearning.COVIDClassification.Configuration.ConfigData;
import ua.univ.distributeddeellearning.COVIDClassification.Configuration.ConfigNetwork;
import ua.univ.distributeddeellearning.COVIDClassification.Evaluator.Evaluation;
import ua.univ.distributeddeellearning.COVIDClassification.Preprocessor.Preprocessor;
import ua.univ.distributeddeellearning.COVIDClassification.Trainer.Trainer;
import ua.univ.distributeddeellearning.COVIDClassification.Utils.ModelUtils;


/**
 * Example class to train and test your model on CPU
 */
public class Classifier {
    private static final Logger log = LoggerFactory.getLogger(Classifier.class);

    public static void main(String[] args) {
        ConfigDLParameters configDLParameters = new ConfigDLParameters();
        ConfigData configData = new ConfigData(configDLParameters);
        Preprocessor preprocessor = new Preprocessor(configDLParameters, configData);
        Evaluation evaluation = new Evaluation(preprocessor, configData);

        if (evaluation.isModelExists()) {
            evaluation.Evaluate();
        } else {
            ConfigNetwork configNetwork = new ConfigNetwork(configDLParameters);
            Trainer trainer = new Trainer(preprocessor, configDLParameters, configData, configNetwork);
            MultiLayerNetwork model = trainer.trainModel();
            ModelUtils.saveModel(model);
        }

        log.info("****************Example finished********************");
    }
}
