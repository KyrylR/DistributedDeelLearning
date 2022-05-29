package ua.univ.distributeddeellearning.COVIDClassification.Configuration;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.LocalResponseNormalization;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Class that's configure MultiLayerNetwork model for
 * classification purpose.
 */
public class ConfigNetwork {
    private final ConfigDLParameters configDLParameters;

    public ConfigNetwork(ConfigDLParameters configDLParameters) {
        this.configDLParameters = configDLParameters;
    }

    /**
     * @return Configured MultiLayerNetwork model
     */
    public MultiLayerNetwork getMultiLayerNetwork() {
        // kernel, stride, pad
        ConvolutionLayer layer0 = new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1},
                new int[]{0, 0}).nIn(this.configDLParameters.getChannel())
                .nOut(50)
                .stride(1, 1)
                .padding(2, 2)
                .name("First convolution layer").activation(Activation.RELU).biasInit(0)
                .build();

        LocalResponseNormalization layer1 = new LocalResponseNormalization.Builder().name("lrn1").build();

        SubsamplingLayer layer2 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(3, 3)
                .stride(2, 2)
                .name("First subsampling layer")
                .build();

        ConvolutionLayer layer3 = new ConvolutionLayer.Builder(5, 5)
                .nIn(50)
                .nOut(70)
                .stride(1, 1)
                .padding(2, 2)
                .name("Second convolution layer")
                .activation(Activation.RELU)
                .biasInit(1)
                .build();

        LocalResponseNormalization layer4 = new LocalResponseNormalization.Builder()
                .name("lrn2")
                .build();

        SubsamplingLayer layer5 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(3, 3)
                .stride(2, 2)
                .name("Second subsampling layer")
                .build();

        ConvolutionLayer layer6 = new ConvolutionLayer.Builder(3, 3)
                .nIn(70)
                .nOut(90)
                .stride(1, 1)
                .padding(2, 2)
                .name("Third convolution layer")
                .activation(Activation.RELU)
                .biasInit(1)
                .build();

        LocalResponseNormalization layer7 = new LocalResponseNormalization.Builder()
                .name("lrn3")
                .build();

        SubsamplingLayer layer8 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(3, 3)
                .stride(2, 2).name("Third subsampling layer").build();


        OutputLayer layer9 = new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                .activation(Activation.SOFTMAX)
                .name("Output").nIn(140).nOut(this.configDLParameters.getOutput())
                .build();

        // Fully Connected Layer Config
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.XAVIER)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(0.006, 0.9))
                .l2(1e-4)
                .miniBatch(false)
                .list()
                .layer(0, layer0)
                .layer(1, layer1)
                .layer(2, layer2)
                .layer(3, layer3)
                .layer(4, layer4)
                .layer(5, layer5)
                .layer(6, layer6)
                .layer(7, layer7)
                .layer(8, layer8)
                .layer(9, layer9)
                .setInputType(InputType.convolutional(
                        this.configDLParameters.getHeight(),
                        this.configDLParameters.getWidth(),
                        this.configDLParameters.getChannel())) // 28 x 28 images having 1 color greyscale
                .backpropType(BackpropType.Standard).build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        model.setListeners(new ScoreIterationListener(20));
        return model;
    }
}
