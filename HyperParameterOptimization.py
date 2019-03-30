'''
HyperParameterOptimization.py
Elliot Trapp
18/11/15

Set of functions that optimize hyperparameters of a prebuilt neural network. The only function that should be called externally is OptimizeHyperParameters.
Should probably use TensorBoard in lieu of this toolset, as it is easier to use, better visualization, and is integrated into TF.
'''

from Libraries.NeuralNetLib.Network import Network
import os

plot_dir=r'./Plots/'

def CallNetFunc(network, func_name, param):

    func_dict={
    'Decay Rate'    : network.SetDecayRate,
    'Momentum'      : network.SetMomentum,
    'Learning Rate' : network.SetLearningRate,
    'Optimizer'     : network.SetOptimizerFunc,
    'Loss Function' : network.SetLossFunc,
    'Batch Size'    : network.SetBatchSize,
    'Num Epochs'    : network.SetNumEpochs,
    'Num Nodes'     : network.SetHiddenNodes,
    'Hidden Layers' : network.SetHiddenLayers
    }

    func_dict[func_name](param)


def PrintResults(metric_result, best_acc, best_score):
    print("{0} with Accuracy of:{1:.2f} and Score of:{2:.2f}".format(metric_result,best_acc,best_score))

def PrintCompleteResults(results):
    for name, metric, acc, score in results:
        PrintResults("Optimal {0}({1:.2f})".format(name,metric),acc,results)

def OptimizeMetric(network,metric_name,metric_list,plot_dir=None):

    optimal_metric = 0.0
    best_acc = 0.0
    best_score = 0.0

    print("Optimizing:\n", metric_name)

    # Save starting weights
    network.model.inner_model.save_weights(r'./my_model_weights.h5')

    for metric in metric_list:

        print("Optimizing {0} with value: {1}".format(metric_name,metric))
        
        # Reset weights each time
        network.model.inner_model.load_weights(r'./my_model_weights.h5')
        
        CallNetFunc(network=network,func_name=metric_name,param=metric)
        #func_dict[metric_name](metric,self=network)
        network.Train()

        title = network.CreatePlotTitle()
        title = title + "_{0}({1})".format(metric_name,metric)
        network.SavePlots(plot_title=title,plot_dir=plot_dir)
        score, acc = network.Evaluate()

        if acc > best_acc: optimal_metric = metric; best_acc = acc; best_score = score

    PrintResults("Optimal {0}({1:.2f})".format(metric_name,optimal_metric),best_acc,best_score)

    os.remove(r'./my_model_weights.h5')

    return optimal_metric, best_acc, best_score

def OptimizeDecayRate(network, decay_rates=None, plot_dir=None):
    """
    Trains network with all rates from decay_rates. Saves and prints the optimal rate at end of iterations
    @param[in] network The network (which includes model) to train on
    @param[in] decay_rates A series of rates to test
    @param[in] save_plots If not None, save plots in this directory
    """

    if decay_rates is None:
        decay_rates = [0.0, 0.001, 0.01, 0.1]

    return OptimizeMetric(network=network,metric_name='Decay Rate',
                          metric_list=decay_rates,plot_dir=plot_dir)


def OptimizeMomentum(network, momentums=None, plot_dir=None):

    if momentums is None:
        momentums = [0.0, 0.001, 0.01, 0.1]

    return OptimizeMetric(network=network,metric_name='Momentum',
                          metric_list=momentums,plot_dir=plot_dir)

def OptimizeLearningRate(network, learning_rates=None, plot_dir=None):

    if learning_rates is None:
        learning_rates = [0.1, 0.01, 0.001]

    return OptimizeMetric(network=network,metric_name='Learning Rate',
                          metric_list=learning_rates,plot_dir=plot_dir)

def OptimizeOptimizerFunc(network, optimizers=None, plot_dir=None):

    if optimizers is None:
        optimizers = ['adam','sgd','rmsprop','adagrad','adadelta','adamax','nadam']

    return OptimizeMetric(network=network,metric_name='Optimizer',
                          metric_list=optimizers,plot_dir=plot_dir)


def OptimizeLossFunc(network, loss_funcs=None, plot_dir=None):

    if loss_funcs is None:
        loss_funcs = ['mean_squared_error','mean_absolute_error','categorical_crossentropy','cosine_proximity']

    return OptimizeMetric(network=network,metric_name='Loss Function',
                          metric_list=loss_funcs,plot_dir=plot_dir)


def OptimizeBatchSize(network, batch_sizes=None, plot_dir=None):

    if batch_sizes is None:
        batch_sizes = [1,16, 32, 64, 100]

    return OptimizeMetric(network=network,metric_name='Batch Size',
                          metric_list=batch_sizes,plot_dir=plot_dir)


def OptimizeNumEpochs(network, epoch_list=None, plot_dir=None):

    if epoch_list is None:
        epoch_list = [5, 15, 30]

    return OptimizeMetric(network=network,metric_name='Num Epochs',
                          metric_list=epoch_list,plot_dir=plot_dir)

def OptimizeNumHiddenNodes(network, num_node_list=None, plot_dir=None):

    if num_node_list is None:
        num_node_list = [32, 64, 128, 512]

    return OptimizeMetric(network=network,metric_name='Num Nodes',
                                               metric_list=num_node_list,plot_dir=plot_dir)


def OptimizeNumHiddenLayers(network, num_layers_list=None, plot_dir=None):

    if num_layers_list is None:
        num_layers_list = [0, 1, 2, 5, 10]

    return OptimizeMetric(network=network,metric_name='Hidden Layers',
                                               metric_list=num_layers_list,plot_dir=plot_dir)


def OptimizeHyperParameters(network, plot_dir=None):

    results = []

#     decay, acc, score = OptimizeDecayRate(network=network,plot_dir=plot_dir)
#     results.append(["Decay Rate", decay, acc, score])

#     mom, acc, score = OptimizeMomentum(network=network,plot_dir=plot_dir)
#     results.append(["Momentum", mom, acc, score])

    lr, acc, score = OptimizeLearningRate(network=network,plot_dir=plot_dir)
    results.append(["Learning Rate", lr, acc, score])

#     optimizer, acc, score = OptimizeOptimizerFunc(network=network,plot_dir=plot_dir)
#     results.append(["Optimizer", optimizer, acc, score])

#     loss, acc, score = OptimizeLossFunc(network=network,plot_dir=plot_dir)
#     results.append(["Loss", loss, acc, score])

    batch_size, acc, score = OptimizeBatchSize(network=network,plot_dir=plot_dir)
    results.append(["Batch Size", batch_size, acc, score])

    epochs, acc, score = OptimizeNumEpochs(network=network,plot_dir=plot_dir)
    results.append(["Epochs", epochs, acc, score])

#     num_layers, acc, score = OptimizeNumHiddenLayers(network=network,plot_dir=plot_dir)
#     results.append(["Num Layers", num_layers, acc, score])

    num_nodes, acc, score = OptimizeNumHiddenNodes(network=network,plot_dir=plot_dir)
    results.append(["Num Nodes", num_nodes, acc, score])

    PrintCompleteResults(results=results)

    print("Optimization complete")