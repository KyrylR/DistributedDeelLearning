package ua.univ.distributeddeellearning.COVIDClassification.Utils;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

/**
 * Some useful utils for MultiLayerNetwork models
 */
public class ModelUtils {
    private static final Logger log = LoggerFactory.getLogger(ModelUtils.class);

    /**
     * @param model MultiLayerNetwork model
     */
    public static void saveModel(MultiLayerNetwork model) {
        String saveDirectory = "src/main/resources/model/";
        log.info("Saving the network and evaluation to directory: {}", saveDirectory);

        try {
            File locationToSave = new File(saveDirectory + "image_classification_model.zip");
            boolean saveUpdater = true;

            ModelSerializer.writeModel(model, locationToSave, saveUpdater);
        } catch (IOException e) {
            log.error(e.getMessage());
        }
    }

}
