package ua.univ.distributeddeellearning.COVIDClassification.MultiGPUtraining;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.nd4j.jita.conf.CudaEnvironment;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ua.univ.distributeddeellearning.COVIDClassification.Configuration.ConfigDLParameters;
import ua.univ.distributeddeellearning.COVIDClassification.Configuration.ConfigData;
import ua.univ.distributeddeellearning.COVIDClassification.Configuration.ConfigNetwork;
import ua.univ.distributeddeellearning.COVIDClassification.Evaluator.Evaluation;
import ua.univ.distributeddeellearning.COVIDClassification.Preprocessor.Preprocessor;
import ua.univ.distributeddeellearning.COVIDClassification.Trainer.Trainer;

/**
 * Example class to train and test your model on multiple GPU(CPU if available)
 */
public class MultiGpuClassificationExample {
    private static final Logger log = LoggerFactory.getLogger(MultiGpuClassificationExample.class);

    public static void main(String[] args) throws Exception {
        configureCdaEnvironment();

        ConfigDLParameters configDLParameters = new ConfigDLParameters();
        ConfigData configData = new ConfigData(configDLParameters);
        Preprocessor preprocessor = new Preprocessor(configDLParameters, configData);
        Evaluation evaluation = new Evaluation(preprocessor, configData);
        ConfigNetwork configNetwork = new ConfigNetwork(configDLParameters);
        Trainer trainer = new Trainer(preprocessor, configDLParameters, configData, configNetwork);

        log.info("Build model....");
        MultiLayerNetwork model = configNetwork.getMultiLayerNetwork();
        ParallelWrapper wrapper = configureGPUsage(model);

        trainer.trainModelGpu(wrapper);

        evaluation.EvaluateModel(model);
    }

    /**
     * ParallelWrapper perform load balancing between GPUs.
     *
     * @param model MultiLayerNetwork model
     * @return wrapper for model
     */
    private static ParallelWrapper configureGPUsage(MultiLayerNetwork model) {
        return new ParallelWrapper.Builder(model)
                // Set it to be equal to number of GPUs on which training is done
                .prefetchBuffer(24)
                // Set to number of physical devices (or x2)
                .workers(2)
                // Rare averaging improves performance, but reduce model accuracy
                .averagingFrequency(3)
                // On every iteration of the model the score will be printed
                .reportScoreAfterAveraging(true)
                .build();
    }

    /**
     * Allow program to use multiple GPU(CPU) if they are available
     */
    private static void configureCdaEnvironment() {
        CudaEnvironment.getInstance().getConfiguration()
                .allowMultiGPU(true)
                // Caches configuration
                .setMaximumDeviceCache(2L * 1024L * 1024L * 1024L)
                // Make faster model averaging over pcie
                .allowCrossDeviceAccess(true);
    }
}
