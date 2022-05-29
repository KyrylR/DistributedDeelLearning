package ua.univ.distributeddeellearning.COVIDClassification.Configuration;

import lombok.Getter;
import lombok.Setter;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;

import java.io.File;
import java.util.Random;

/**
 * Data files path configuration
 */
@Getter
@Setter
public class ConfigData {
    /**
     * Data train path -> data will be used to feed the model during training
     */
    private final File trainPath;

    /**
     * Data test path -> data will be used to evaluate the model for each iteration during training
     */
    private final File testPath;

    /**
     * Data evaluation path -> data will be used to evaluate the model after the model has been built
     */
    private final File evaluationPath;

    private final Random randNumGen;

    public ConfigData(ConfigDLParameters configDLParameters) {
        this.trainPath = new File("C:\\Users\\inter\\Desktop\\data\\train\\");
        this.testPath = new File("C:\\Users\\inter\\Desktop\\data\\test\\");
        this.evaluationPath = new File("C:\\Users\\inter\\Desktop\\data\\evaluation\\");
        randNumGen = configDLParameters.getRandNumGen();
    }

    public FileSplit getTrainPath() {
        return new FileSplit(this.trainPath, NativeImageLoader.ALLOWED_FORMATS, this.randNumGen);
    }

    public FileSplit getTestPath() {
        return new FileSplit(this.testPath, NativeImageLoader.ALLOWED_FORMATS, this.randNumGen);
    }

    public FileSplit getEvaluationPath() {
        return new FileSplit(this.evaluationPath, NativeImageLoader.ALLOWED_FORMATS, this.randNumGen);
    }
}
