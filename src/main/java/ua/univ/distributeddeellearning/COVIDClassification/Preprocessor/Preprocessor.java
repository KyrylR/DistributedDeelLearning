package ua.univ.distributeddeellearning.COVIDClassification.Preprocessor;

import lombok.Getter;
import lombok.Setter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ua.univ.distributeddeellearning.COVIDClassification.Configuration.ConfigDLParameters;
import ua.univ.distributeddeellearning.COVIDClassification.Configuration.ConfigData;

/**
 * Class that used for preprocessing data and
 * creation it's data set iterator
 */
@Getter
@Setter
public class Preprocessor {
    private static final Logger log = LoggerFactory.getLogger(Preprocessor.class);
    private final ParentPathLabelGenerator labelMaker;
    private ConfigDLParameters configDLParameters;
    private ConfigData configData;

    public Preprocessor(ConfigDLParameters configDLParameters, ConfigData configData) {
        this.configDLParameters = configDLParameters;
        this.configData = configData;
        this.labelMaker = new ParentPathLabelGenerator();
    }

    /**
     * @param data files
     * @return data set iterator
     */
    public DataSetIterator getDataSetIterator(FileSplit data) {
        try (ImageRecordReader preprocessRR = new ImageRecordReader(
                this.configDLParameters.getHeight(),
                this.configDLParameters.getWidth(),
                this.configDLParameters.getChannel(), labelMaker)) {
            preprocessRR.initialize(data);
            DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(
                    preprocessRR, this.configDLParameters.getBatchSize(),
                    1, this.configDLParameters.getOutput());
            DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
            scaler.fit(dataSetIterator);
            dataSetIterator.setPreProcessor(scaler);
            return dataSetIterator;
        } catch (Exception exception) {
            log.error(exception.getMessage());
        }
        return null;
    }
}
