package ua.univ.distributeddeellearning.COVIDClassification.Configuration;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;

import java.util.Random;

/**
 * Config basic parameters for DeepLearning, such as
 * height, width, channel, random seed, batch size,
 * output layers, epoch number.
 */
@Getter
@Setter
@Slf4j
@AllArgsConstructor
public class ConfigDLParameters {
    private int height;
    private int width;
    private int channel;
    private int seed;
    private int batchSize;
    private int output;
    private int epoch;

    public ConfigDLParameters() {
        this.height = 28;
        this.width = 28;
        // Grayscale
        this.channel = 1;
        this.seed = 12345;
        // Number of data to be trained for each iteration
        this.batchSize = 50;
        this.output = 2;
        this.epoch = 350;
    }

    public ConfigDLParameters(int height, int width, int channel) {
        this.height = height;
        this.width = width;
        this.channel = channel;
    }

    /**
     * @return random number generator
     */
    public Random getRandNumGen() {
        return new Random(this.seed);
    }
}
