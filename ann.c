#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double sigmoid(double x);
double dSigmoid(double x);
double initWeight();
void shuffle(int *array, int n);
double loss(double real, double pred);



#define numHiddenNodes 2
#define numOutputs 1
#define numInputs 2
#define lr 0.1
#define numTrainingSets 4

void main(){

    double hiddenLayer[numHiddenNodes];
    double outputLayer[numOutputs];

    double hiddenLayerBias[numHiddenNodes];
    double outputLayerBias[numOutputs];

    double hiddenWeights[numInputs][numHiddenNodes];
    double outputWeights[numHiddenNodes][numOutputs];

    double training_inputs[numTrainingSets][numInputs] = { {0.0,0.0},{1.0,0.0},{0.0,1.0},{1.0,1.0} };
    double training_outputs[numTrainingSets][numOutputs] = { {0.0},{1.0},{1.0},{0.0} };

    //initialize hidden weight
    for (int i=0; i<numInputs; i++) {
        for (int j=0; j<numHiddenNodes; j++) {
            hiddenWeights[i][j] = initWeight();
        }
    }

    //initialize output weight and biases
    for (int i=0; i<numHiddenNodes; i++) {
        hiddenLayerBias[i] = initWeight();
        for (int j=0; j<numOutputs; j++) {
            outputLayerBias[j] = initWeight();
            outputWeights[i][j] = initWeight();
        }
    }

    int trainingSetOrder[] = {0,1,2,3};

    int num_iter = 1000;
    int display_step = 50;

    double error_n = 0;

    for (int i=1; i < num_iter+1; i++){
        shuffle(trainingSetOrder, numTrainingSets);

        for (int j=0; j < numTrainingSets; j++){

            int x = trainingSetOrder[j];

            //Forward propagation

            // input layer forward propagation with sigmoid activation
            for (int k=0; k < numHiddenNodes; k++){
                double bias=hiddenLayerBias[k];
                for (int l=0; l<numInputs; l++) {
                    bias+= training_inputs[x][l] * hiddenWeights[l][k];
                }
                hiddenLayer[k] = sigmoid(bias);
            }

            // hidden layer forward propagation with sigmoid activation

            for (int k=0; k<numOutputs; k++) {
                double activation=outputLayerBias[k];
                for (int l=0; l<numHiddenNodes; l++) {
                    activation+=hiddenLayer[l]*outputWeights[l][k];
                }
                outputLayer[k] = sigmoid(activation);
            }

            error_n += loss(training_outputs[x][0], outputLayer[0]);
            if (i % display_step == 0){
//                printf("Step %d, Error: %.4f, Inputs: [%f][%f], Pred: [%.4f], Out: [%f] \n",
//                        i, error_n/ i, training_inputs[x][0], training_inputs[x][1], outputLayer[0], training_outputs[x][0]);
                    printf("Step %d, Error: %.4f \n", i, error_n/ i);
            }

            // Forward propagation complete

            // Back propagation using gradient descent

            // loss calculations

            double deltaOutput[numOutputs];
            for (int k=0; k < numOutputs; k++){
                double errorOutput = loss(training_outputs[x][k], outputLayer[k]);
                deltaOutput[k] = errorOutput* dSigmoid(outputLayer[k]);
            }

            double deltaHidden[numHiddenNodes];
            for (int k=0; k<numHiddenNodes; k++) {
                double errorHidden = 0.0;
                for(int l=0; l<numOutputs; l++) {
                    errorHidden+=deltaOutput[l]* outputWeights[k][l];
                }
                deltaHidden[k] = errorHidden* dSigmoid(hiddenLayer[k]);
            }

            // loss propagation

            for (int k=0; k<numOutputs; k++) {
                outputLayerBias[k] += deltaOutput[k]* lr;
                for (int l=0; l<numHiddenNodes; l++) {
                    outputWeights[l][k]+= hiddenLayer[k]*deltaOutput[k]*lr;
                }
            }

            for (int k=0; k<numHiddenNodes; k++) {
                hiddenLayerBias[k] += deltaHidden[k]* lr;
                for (int l=0; l<numInputs; l++) {
                    hiddenWeights[l][k]+= training_inputs[x][k]*deltaHidden[k]*lr;
                }
            }


        }
    }
}


double sigmoid(double x) {
    // activation function
    return 1 / (1 + exp(-x));
}
double dSigmoid(double x) {
    // derivative of sigmoid
    return x * (1 - x);
}

double initWeight() {
    // return a uniformly distributed random value
    return ( (double)(rand()) + 1. )/( (double)(RAND_MAX) + 1. );
}

double loss(double real, double pred){
    return real - pred;
    /*
    if ((int) real == 1)
        return -log(pred);
    return -log(1-pred);
     */
}

void shuffle(int *array, int n)
{
    // shuffle array of int datatype
    //TODO: make it work with any datatype
    if (n > 1)
    {
        int i;
        for (i = 0; i < n - 1; i++)
        {
            int j =  (i + rand() / (RAND_MAX / (n - i) + 1));
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}




