package ua.univ.distributeddeellearning.COVIDClassification.Trainer;


import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ua.univ.distributeddeellearning.COVIDClassification.Configuration.ConfigDLParameters;
import ua.univ.distributeddeellearning.COVIDClassification.Configuration.ConfigData;
import ua.univ.distributeddeellearning.COVIDClassification.Configuration.ConfigNetwork;
import ua.univ.distributeddeellearning.COVIDClassification.Preprocessor.Preprocessor;

import java.io.FileWriter;

/**
 * Class for training a model on a CPU or GPU
 */
public class Trainer {
    private static final Logger log = LoggerFactory.getLogger(Trainer.class);
    private final Preprocessor preprocessor;
    private final ConfigDLParameters configDLParameters;
    private final ConfigData configData;
    private final ConfigNetwork configNetwork;

    public Trainer(Preprocessor preprocessor,
                   ConfigDLParameters configDLParameters,
                   ConfigData configData,
                   ConfigNetwork configNetwork) {

        this.preprocessor = preprocessor;
        this.configDLParameters = configDLParameters;
        this.configData = configData;
        this.configNetwork = configNetwork;
    }

    /**
     * @return MultiLayerNetwork trained model
     */
    public MultiLayerNetwork trainModel() {
        log.info("Train model....");
        long timeX = System.currentTimeMillis();

        String modelLogPath = "src/main/resources/model/image_classification_model_log.txt";
        try (FileWriter fileWriter = new FileWriter(modelLogPath)) {
            DataSetIterator trainIter = preprocessor.getDataSetIterator(configData.getTrainPath());
            DataSetIterator testIter = preprocessor.getDataSetIterator(configData.getTestPath());
            MultiLayerNetwork model = configNetwork.getMultiLayerNetwork();
            for (int i = 0; i < this.configDLParameters.getEpoch(); i++) {
                trainIter.reset();
                long time1 = System.currentTimeMillis();
                while (trainIter.hasNext()) {
                    model.fit(trainIter.next());
                }
                long time2 = System.currentTimeMillis();
                Evaluation eval = model.evaluate(testIter);
                log.info("*** Completed epoch: {}, time: {} ***", i, (time2 - time1));
                String stats = eval.stats();
                log.info("{}", stats);
                fileWriter.write(String.format("%nCompleted epoch: %d;%n %s", i, stats));
            }
            long timeY = System.currentTimeMillis();

            log.info("*** Training complete, time: {} ***", (timeY - timeX));
            return model;
        } catch (Exception exception) {
            log.error(exception.getMessage());
        }
        return null;
    }

    /**
     * Training model with multi GPU(or CPU)
     *
     * @param wrapper GPU wrapper
     */
    public void trainModelGpu(ParallelWrapper wrapper) {
        log.info("Train model with GPU....");
        long timeX = System.currentTimeMillis();
        DataSetIterator trainIter = preprocessor.getDataSetIterator(configData.getTrainPath());

        for (int i = 0; i < configDLParameters.getEpoch(); i++) {
            long time1 = System.currentTimeMillis();
            // fit on ParallelWrapper not the model.fit. ParallelWrapper will call or model underneath
            wrapper.fit(trainIter);
            long time2 = System.currentTimeMillis();
            log.info("*** Completed epoch: {}, time: {} ***", i, (time2 - time1));
        }
        long timeY = System.currentTimeMillis();

        log.info("*** Training complete, time: {} ***", (timeY - timeX));
    }
}
