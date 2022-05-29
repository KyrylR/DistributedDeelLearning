package ua.univ.distributeddeellearning.COVIDClassification.DistributedClassifier;

import com.beust.jcommander.Parameter;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.LossLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.zoo.model.helper.DarknetHelper;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.AMSGrad;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ua.univ.distributeddeellearning.COVIDClassification.Configuration.ConfigDLParameters;
import ua.univ.distributeddeellearning.COVIDClassification.Configuration.ConfigData;
import ua.univ.distributeddeellearning.COVIDClassification.Preprocessor.Preprocessor;
import ua.univ.distributeddeellearning.COVIDClassification.Utils.JCommanderUtils;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;

/**
 * This is a local (single-machine) version of the Distributed image classifier.
 */
public class TrainLocal {
    public static Logger log = LoggerFactory.getLogger(TrainLocal.class);

    @Parameter(names = {"--numEpochs"}, description = "Number of epochs for training")
    private final int numEpochs = 10;

    @Parameter(names = {"--saveDir"}, description = "If set, the directory to save the trained network")
    private final String saveDir = "src/main/resources/result";

    public static void main(String[] args) throws Exception {
        new TrainLocal().entryPoint(args);
    }

    /**
     * This network: created for the purposes of this example.
     * It is a simple CNN loosely inspired by the DarkNet
     * architecture, which was in turn inspired by the VGG16/19 networks
     * The performance of this network can likely be improved
     *
     * @return Computation Graph network
     */
    public static ComputationGraph getNetwork() {
        ISchedule lrSchedule = new MapSchedule.Builder(ScheduleType.EPOCH)
                .add(0, 8e-3)
                .add(1, 6e-3)
                .add(3, 3e-3)
                .add(5, 1e-3)
                .add(7, 5e-4).build();

        ComputationGraphConfiguration.GraphBuilder b = new NeuralNetConfiguration.Builder()
                .convolutionMode(ConvolutionMode.Same)
                .l2(1e-4)
                .updater(new AMSGrad(lrSchedule))
                .weightInit(WeightInit.RELU)
                .graphBuilder()
                .addInputs("input")
                .setOutputs("output");

        DarknetHelper.addLayers(b, 0, 3, 3, 32, 0);     //64x64 out
        DarknetHelper.addLayers(b, 1, 3, 32, 64, 2);    //32x32 out
        DarknetHelper.addLayers(b, 2, 2, 64, 128, 0);   //32x32 out
        DarknetHelper.addLayers(b, 3, 2, 128, 256, 2);   //16x16 out
        DarknetHelper.addLayers(b, 4, 2, 256, 256, 0);   //16x16 out
        DarknetHelper.addLayers(b, 5, 2, 256, 512, 2);   //8x8 out

        b.addLayer("convolution2d_6", new ConvolutionLayer.Builder(6, 6)
                        .nIn(1)
                        .nOut(20)
                        //.weightInit(WeightInit.XAVIER)
                        .stride(2, 2)
                        .activation(Activation.RELU)
                        .build(), "maxpooling2d_5")
                .addLayer("globalpooling", new GlobalPoolingLayer.Builder(PoolingType.AVG).build(), "convolution2d_6")
                .addLayer("loss", new LossLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX).build(), "globalpooling")
                .setOutputs("loss");

        ComputationGraphConfiguration conf = b.build();

        ComputationGraph net = new ComputationGraph(conf);
        net.init();

        return net;
    }

    /**
     * @param args
     */
    public void entryPoint(String[] args) {
        JCommanderUtils.parseArgs(this, args);

        ConfigDLParameters configDLParameters = new ConfigDLParameters();
        ConfigData configData = new ConfigData(configDLParameters);
        Preprocessor preprocessor = new Preprocessor(configDLParameters, configData);

        //Create the data pipeline
        DataSetIterator iter = preprocessor.getDataSetIterator(configData.getTrainPath());

        //Create the network
        ComputationGraph net = getNetwork();
        net.setListeners(new PerformanceListener(50, true));

        //Reduce auto GC frequency for better performance
        Nd4j.getMemoryManager().setAutoGcWindow(10000);

        //Fit the network
        net.fit(iter, numEpochs);
        log.info("Training complete. Starting evaluation.");

        //Evaluate the network on test set data
        DataSetIterator test = preprocessor.getDataSetIterator(configData.getTestPath());
        Evaluation e = new Evaluation();
        net.doEvaluation(test, e);

        log.info("Evaluation complete");
        log.info(e.stats());

        File sd = new File(saveDir);
        if (!sd.exists())
            sd.mkdirs();

        log.info("Saving network and evaluation stats to directory: {}", saveDir);

        try {
            net.save(new File(saveDir, "trainedNet.bin"));
            FileUtils.writeStringToFile(new File(saveDir, "evaulation.txt"), e.stats(), StandardCharsets.UTF_8);
        } catch (IOException ex) {
            log.error(ex.getMessage());
        }

        log.info("----- Examples Complete -----");
    }
}
