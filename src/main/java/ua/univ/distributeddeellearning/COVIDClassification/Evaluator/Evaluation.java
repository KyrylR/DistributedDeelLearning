package ua.univ.distributeddeellearning.COVIDClassification.Evaluator;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ua.univ.distributeddeellearning.COVIDClassification.Configuration.ConfigData;
import ua.univ.distributeddeellearning.COVIDClassification.Preprocessor.Preprocessor;

import java.io.File;
import java.io.FileWriter;
import java.util.Arrays;
import java.util.List;

/**
 * Class for model evaluation.
 */
public class Evaluation {
    private static final Logger log = LoggerFactory.getLogger(Evaluation.class);
    private static final String MODEL_PATH = "src/main/resources/model/image_classification_model.zip";
    private final Preprocessor preprocessor;
    private final ConfigData configData;

    public Evaluation(Preprocessor preprocessor, ConfigData configData) {
        this.preprocessor = preprocessor;
        this.configData = configData;
    }

    /**
     * Evaluation on existing resource.
     */
    public void Evaluate() {
        log.info("Model found!\nLoad model.....");

        String evaluationResult = "src/main/resources/result/evaluation_output.txt";
        try (FileWriter fileWriter = new FileWriter(evaluationResult)) {

            // Set params for evaluation data
            MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(MODEL_PATH);
            model.getLabels();

            DataSetIterator evalIter = preprocessor.getDataSetIterator(configData.getEvaluationPath());

            List<String> labelList = Arrays.asList("Normal", "COVID");

            // Example on how to get predict results with trained model
            evalIter.reset();
            DataSet evalDataSet = evalIter.next();
            evalDataSet.setLabelNames(labelList);
            String expectedResult = evalDataSet.getLabelName(0);
            List<String> predict = model.predict(evalDataSet);
            String modelResult = predict.get(0); //Predict the first image (image at index 0) inside
            //target/classes/dataset/evaluation/bee folder,
            //change or replace the images in this folder with bee or spider image,
            //just if you want to.

            String evaluationOutput = "\nFor a single example that is labeled " + expectedResult
                    + " the model predicted as " + modelResult + "\n\n";

            log.info(evaluationOutput);
            fileWriter.write(evaluationOutput);
        } catch (Exception exception) {
            log.error(exception.getMessage());
        }
    }

    /**
     * Evaluation on existing resource.
     *
     * @param model MultiLayerNetwork custom model
     */
    public void EvaluateModel(MultiLayerNetwork model) {
        log.info("Model found!\nLoad model.....");

        String evaluationResult = "src/main/resources/result/evaluation_output.txt";
        try (FileWriter fileWriter = new FileWriter(evaluationResult)) {
            DataSetIterator evalIter = preprocessor.getDataSetIterator(configData.getEvaluationPath());

            List<String> labelList = Arrays.asList("Normal", "COVID");

            // Example on how to get predict results with trained model
            evalIter.reset();
            DataSet evalDataSet = evalIter.next();
            evalDataSet.setLabelNames(labelList);
            String expectedResult = evalDataSet.getLabelName(0);
            List<String> predict = model.predict(evalDataSet);
            String modelResult = predict.get(0); //Predict the first image (image at index 0) inside
            //target/classes/dataset/evaluation/bee folder,
            //change or replace the images in this folder with bee or spider image,
            //just if you want to.

            String evaluationOutput = "\nFor a single example that is labeled " + expectedResult
                    + " the model predicted as " + modelResult + "\n\n";

            log.info(evaluationOutput);
            fileWriter.write(evaluationOutput);
        } catch (Exception exception) {
            log.error(exception.getMessage());
        }
    }

    public boolean isModelExists() {
        File file = new File(MODEL_PATH);
        return file.exists();
    }
}
