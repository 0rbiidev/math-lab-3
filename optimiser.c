#include "optimiser.h"
#include "mnist_helper.h"
#include "neural_network.h"
#include "math.h"
#include <time.h>
#include <stdlib.h>

// Function declarations
void update_parameters(unsigned int batch_size);
void print_training_stats(unsigned int epoch_counter, unsigned int total_iter, double mean_loss, double test_accuracy);

// Optimisation parameters
unsigned int log_freq = 30000; // Compute and print accuracy every log_freq iterations

double alpha_decay = 0.9; //used for momentum

// Parameters passed from command line arguments
unsigned int num_batches;
unsigned int batch_size;
unsigned int total_epochs;
double learning_rate;

void print_training_stats(unsigned int epoch_counter, unsigned int total_iter, double mean_loss, double test_accuracy){
    printf("Epoch: %u,  Total iter: %u,  Mean Loss: %0.12f,  Test Acc: %f\n", epoch_counter, total_iter, mean_loss, test_accuracy);
}

void initialise_optimiser(double cmd_line_learning_rate, int cmd_line_batch_size, int cmd_line_total_epochs){
    batch_size = cmd_line_batch_size;
    learning_rate = cmd_line_learning_rate;
    total_epochs = cmd_line_total_epochs;
    
    num_batches = total_epochs * (N_TRAINING_SET / batch_size);
    printf("Optimising with parameters: \n\tepochs = %u \n\tbatch_size = %u \n\tnum_batches = %u\n\tlearning_rate = %f\n\n",
           total_epochs, batch_size, num_batches, learning_rate);
    
    srand(time(NULL));
}

double lr_decay(double init, double final, double ecounter, double etotal){ // written for p2
    return init*(1 - (ecounter/etotal)) - (ecounter/etotal)*final;
}

void momentum_update(unsigned int batch_size){ // written for p2
    for (int i = 0; i < N_NEURONS_LI; i++){
        for (int j = 0; j < N_NEURONS_L1; j++){
            double grd = w_LI_L1[i][j].dw * (learning_rate / (double)batch_size);
            w_LI_L1[i][j].v = alpha_decay*w_LI_L1[i][j].v - (1-alpha_decay)*grd;
            w_LI_L1[i][j].w += w_LI_L1[i][j].v;
            w_LI_L1[i][j].dw = 0;
        }
    }
    for (int i = 0; i < N_NEURONS_L1; i++){
        for (int j = 0; j < N_NEURONS_L2; j++){
            double grd = w_L1_L2[i][j].dw * (learning_rate / (double)batch_size);
            w_L1_L2[i][j].v = alpha_decay*w_L1_L2[i][j].v - (1-alpha_decay)*grd;
            w_L1_L2[i][j].w += w_L1_L2[i][j].v;
            w_L1_L2[i][j].dw = 0;
        }
    }
    for (int i = 0; i < N_NEURONS_L2; i++){
        for (int j = 0; j < N_NEURONS_L3; j++){
            double grd = w_L2_L3[i][j].dw * (learning_rate / (double)batch_size);
            w_L2_L3[i][j].v = alpha_decay*w_L2_L3[i][j].v - (1-alpha_decay)*grd;
            w_L2_L3[i][j].w += w_L2_L3[i][j].v;
            w_L2_L3[i][j].dw = 0;
        }
    }
}

void adagrad_update(unsigned int batch_size){ // written for p3
    // adagrad constant delta = __DBL_MIN__ - smallest double > 0

    for (int i = 0; i < N_NEURONS_LI; i++){
        for (int j = 0; j < N_NEURONS_L1; j++){
            double grd = w_LI_L1[i][j].dw/(double)batch_size;
            w_LI_L1[i][j].p += pow(grd, 2);
            w_LI_L1[i][j].w -= (learning_rate / (__DBL_MIN__ + sqrt(w_LI_L1[i][j].p))) * grd;
            w_LI_L1[i][j].dw = 0;
        }
    }
    for (int i = 0; i < N_NEURONS_L1; i++){
        for (int j = 0; j < N_NEURONS_L2; j++){
            double grd = w_L1_L2[i][j].dw/(double)batch_size;
            w_L1_L2[i][j].p += pow(grd, 2);
            w_L1_L2[i][j].w -= (learning_rate / (__DBL_MIN__ + sqrt(w_L1_L2[i][j].p))) * grd;
            w_L1_L2[i][j].dw = 0;
        }
    }
    for (int i = 0; i < N_NEURONS_L2; i++){
        for (int j = 0; j < N_NEURONS_L3; j++){
            double grd = w_L2_L3[i][j].dw/(double)batch_size;
            w_L2_L3[i][j].p += pow(grd, 2);
            w_L2_L3[i][j].w -= (learning_rate / (__DBL_MIN__ + sqrt(w_L2_L3[i][j].p))) * grd;
            w_L2_L3[i][j].dw = 0;
        }
    }
}


void run_optimisation(void){
    unsigned int training_sample = 0;
    unsigned int total_iter = 0;
    double obj_func = 0.0;
    unsigned int epoch_counter = 0;
    double test_accuracy = 0.0;  //evaluate_testing_accuracy();
    double mean_loss = 0.0;
    double learning_rate_init = learning_rate;
    double learning_rate_fin = 0.00001;
    
    // Run optimiser - update parameters after each minibatch
    for (int i=0; i < num_batches; i++){
        for (int j = 0; j < batch_size; j++){

            // Evaluate accuracy on testing set (expensive, evaluate infrequently)
            if (total_iter % log_freq == 0 || total_iter == 0){
                if (total_iter > 0){
                    mean_loss = mean_loss/((double) log_freq);
                }
                
                test_accuracy = evaluate_testing_accuracy();
                print_training_stats(epoch_counter, total_iter, mean_loss, test_accuracy);

                // Reset mean_loss for next reporting period
                mean_loss = 0.0;
            }
            
            // Evaluate forward pass and calculate gradients
            obj_func = evaluate_objective_function(training_sample);
            mean_loss+=obj_func;

            // Update iteration counters (reset at end of training set to allow multiple epochs)
            total_iter++;
            training_sample++;
            // On epoch completion:
            if (training_sample == N_TRAINING_SET){
                training_sample = 0;
                epoch_counter++;

                // learning rate decay
                //learning_rate = lr_decay(learning_rate_init, learning_rate_fin, epoch_counter, total_epochs);
            }
        }
        
        // Update weights on batch completion

        //update_parameters(batch_size); //p1
        //momentum_update(batch_size); //p2
        adagrad_update(batch_size); //p3
    }
    
    // Print final performance
    test_accuracy = evaluate_testing_accuracy();
    print_training_stats(epoch_counter, total_iter, (mean_loss/((double) log_freq)), test_accuracy);
}

double evaluate_objective_function(unsigned int sample){

    // Compute network performance
    evaluate_forward_pass(training_data, sample);
    double loss = compute_xent_loss(training_labels[sample]);
    
    // Evaluate gradients
    //evaluate_backward_pass(training_labels[sample], sample);
    evaluate_backward_pass_sparse(training_labels[sample], sample);
    
    // Evaluate parameter updates
    store_gradient_contributions();
    
    return loss;
}

void update_parameters(unsigned int batch_size){
    // Part I To-do

    for (int i = 0; i < N_NEURONS_LI; i++){
        for (int j = 0; j < N_NEURONS_L1; j++){
            double dw = -(learning_rate * w_LI_L1[i][j].dw);
            w_LI_L1[i][j].w += dw / (double)batch_size;
            w_LI_L1[i][j].dw = 0;
        }
    }

    for (int i = 0; i < N_NEURONS_L1; i++){
        for (int j = 0; j < N_NEURONS_L2; j++){
            double dw = -(learning_rate * w_L1_L2[i][j].dw);
            w_L1_L2[i][j].w += dw / (double)batch_size;
            w_L1_L2[i][j].dw = 0;
        }
    }

    for (int i = 0; i < N_NEURONS_L2; i++){
        for (int j = 0; j < N_NEURONS_L3; j++){
            double dw = -(learning_rate * w_L2_L3[i][j].dw);
            w_L2_L3[i][j].w += dw / (double)batch_size;
            w_L2_L3[i][j].dw = 0;
        }
    }
}


