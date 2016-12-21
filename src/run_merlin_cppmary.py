#!/usr/bin/python
#coding=utf-8
import cPickle
import gzip
import os, sys, errno
import time
import math

import subprocess
import socket # only for socket.getfqdn()

#  numpy & theano imports need to be done in this order (only for some numpy installations, not sure why)
import numpy
#import gnumpy as gnp
# we need to explicitly import this in some cases, not sure why this doesn't get imported with numpy itself
import numpy.distutils.__config__
# and only after that can we import theano
import theano

from utils.providers import ListDataProvider

from frontend.label_normalisation import HTSLabelNormalisation, XMLLabelNormalisation
from frontend.label_cppmary import LabelCppmary
from frontend.silence_remover import SilenceRemover
from frontend.silence_remover import trim_silence
from frontend.min_max_norm import MinMaxNormalisation
from frontend.acoustic_composition import AcousticComposition
from frontend.parameter_generation import ParameterGeneration
from frontend.mean_variance_norm import MeanVarianceNorm

# the new class for label composition and normalisation
from frontend.label_composer import LabelComposer
from frontend.label_modifier import HTSLabelModification
#from frontend.mlpg_fast import MLParameterGenerationFast

#from frontend.mlpg_fast_layer import MLParameterGenerationFastLayer


import configuration
from models.deep_rnn import DeepRecurrentNetwork

from utils.compute_distortion import DistortionComputation, IndividualDistortionComp
from utils.generate import generate_wav
from utils.learn_rates import ExpDecreaseLearningRate

from io_funcs.binary_io import  BinaryIOCollection

#import matplotlib.pyplot as plt
# our custom logging class that can also plot
#from logplot.logging_plotting import LoggerPlotter, MultipleTimeSeriesPlot, SingleWeightMatrixPlot
from logplot.logging_plotting import LoggerPlotter, MultipleSeriesPlot, SingleWeightMatrixPlot
import logging # as logging
import logging.config
import StringIO
import random
import mxnet as mx
from mxnet_merlin import *


def extract_file_id_list(file_list):
    file_id_list = []
    for file_name in file_list:
        file_id = os.path.basename(os.path.splitext(file_name)[0])
        file_id_list.append(file_id)

    return  file_id_list

def read_file_list(file_name):

    logger = logging.getLogger("read_file_list")

    file_lists = []
    fid = open(file_name)
    for line in fid.readlines():
        line = line.strip()
        if len(line) < 1:
            continue
        file_lists.append(line)
    fid.close()

    logger.debug('Read file list from %s' % file_name)
    return  file_lists


def make_output_file_list(out_dir, in_file_lists):
    out_file_lists = []

    for in_file_name in in_file_lists:
        file_id = os.path.basename(in_file_name)
        out_file_name = out_dir + '/' + file_id
        out_file_lists.append(out_file_name)

    return  out_file_lists

def prepare_file_path_list(file_id_list, file_dir, file_extension, new_dir_switch=True):
    if not os.path.exists(file_dir) and new_dir_switch:
        os.makedirs(file_dir)
    file_name_list = []
    for file_id in file_id_list:
        file_name = file_dir + '/' + file_id + file_extension
        file_name_list.append(file_name)

    return  file_name_list



def visualize_dnn(dnn):

    layer_num = len(dnn.params)     ## including input and output
    plotlogger = logging.getLogger("plotting")

    for i in xrange(layer_num):
        fig_name = 'Activation weights W' + str(i) + '_' + dnn.params[i].name
        fig_title = 'Activation weights of W' + str(i)
        xlabel = 'Neuron index of hidden layer ' + str(i)
        ylabel = 'Neuron index of hidden layer ' + str(i+1)
        if i == 0:
            xlabel = 'Input feature index'
        if i == layer_num-1:
            ylabel = 'Output feature index'

        aa = dnn.params[i].get_value(borrow=True).T
        print   aa.shape, aa.size
        if aa.size > aa.shape[0]:
            logger.create_plot(fig_name, SingleWeightMatrixPlot)
            plotlogger.add_plot_point(fig_name, fig_name, dnn.params[i].get_value(borrow=True).T)
            plotlogger.save_plot(fig_name, title=fig_name, xlabel=xlabel, ylabel=ylabel)

def load_covariance(var_file_dict, out_dimension_dict):
    var = {}
    io_funcs = BinaryIOCollection()
    for feature_name in var_file_dict.keys():
        var_values, dimension = io_funcs.load_binary_file_frame(var_file_dict[feature_name], 1)

        var_values = numpy.reshape(var_values, (out_dimension_dict[feature_name], 1))

        var[feature_name] = var_values

    return  var


def train_DNN(train_xy_file_list, valid_xy_file_list, \
              nnets_file_name, n_ins, n_outs, ms_outs, hyper_params, buffer_size, plot=False, var_dict=None,
              cmp_mean_vector = None, cmp_std_vector = None, init_dnn_model_file = None):

    # get loggers for this function
    # this one writes to both console and file
    logger = logging.getLogger("main.train_DNN")
    logger.debug('Starting train_DNN')

    if plot:
        # this one takes care of plotting duties
        plotlogger = logging.getLogger("plotting")
        # create an (empty) plot of training convergence, ready to receive data points
        logger.create_plot('training convergence',MultipleSeriesPlot)

    try:
        assert numpy.sum(ms_outs) == n_outs
    except AssertionError:
        logger.critical('the summation of multi-stream outputs does not equal to %d' %(n_outs))
        raise

    ####parameters#####
    finetune_lr     = float(hyper_params['learning_rate'])
    training_epochs = int(hyper_params['training_epochs'])
    batch_size      = int(hyper_params['batch_size'])
    l1_reg          = float(hyper_params['l1_reg'])
    l2_reg          = float(hyper_params['l2_reg'])
    warmup_epoch    = int(hyper_params['warmup_epoch'])
    momentum        = float(hyper_params['momentum'])
    warmup_momentum = float(hyper_params['warmup_momentum'])

    hidden_layer_size = hyper_params['hidden_layer_size']

    buffer_utt_size = buffer_size
    early_stop_epoch = int(hyper_params['early_stop_epochs'])

    hidden_activation = hyper_params['hidden_activation']
    output_activation = hyper_params['output_activation']

    model_type = hyper_params['model_type']
    hidden_layer_type  = hyper_params['hidden_layer_type']

    ## use a switch to turn on pretraining
    ## pretraining may not help too much, if this case, we turn it off to save time
    do_pretraining = hyper_params['do_pretraining']
    pretraining_epochs = int(hyper_params['pretraining_epochs'])
    pretraining_lr = float(hyper_params['pretraining_lr'])

    sequential_training = hyper_params['sequential_training']
    dropout_rate = hyper_params['dropout_rate']

#    sequential_training = True

    buffer_size = int(buffer_size / batch_size) * batch_size

    ###################
    (train_x_file_list, train_y_file_list) = train_xy_file_list
    (valid_x_file_list, valid_y_file_list) = valid_xy_file_list

    logger.debug('Creating training   data provider')
    train_data_reader = ListDataProvider(x_file_list = train_x_file_list, y_file_list = train_y_file_list,
                            n_ins = n_ins, n_outs = n_outs, buffer_size = buffer_size, sequential = sequential_training, shuffle = True)

    logger.debug('Creating validation data provider')
    valid_data_reader = ListDataProvider(x_file_list = valid_x_file_list, y_file_list = valid_y_file_list,
                            n_ins = n_ins, n_outs = n_outs, buffer_size = buffer_size, sequential = sequential_training, shuffle = False)

    shared_train_set_xy, temp_train_set_x, temp_train_set_y = train_data_reader.load_one_partition()
    train_set_x, train_set_y = shared_train_set_xy
    shared_valid_set_xy, valid_set_x, valid_set_y = valid_data_reader.load_one_partition()   #validation data is still read block by block
    valid_set_x, valid_set_y = shared_valid_set_xy
    train_data_reader.reset()
    valid_data_reader.reset()


    ##temporally we use the training set as pretrain_set_x.
    ##we need to support any data for pretraining
#    pretrain_set_x = train_set_x

    # numpy random generator
    numpy_rng = numpy.random.RandomState(123)
    logger.info('building the model')


    dnn_model = None
    pretrain_fn = None  ## not all the model support pretraining right now
    train_fn = None
    valid_fn = None
    valid_model = None ## valid_fn and valid_model are the same. reserve to computer multi-stream distortion
    if model_type == 'DNN':
        dnn_model = DeepRecurrentNetwork(n_in= n_ins, hidden_layer_size = hidden_layer_size, n_out = n_outs,
                                         L1_reg = l1_reg, L2_reg = l2_reg, hidden_layer_type = hidden_layer_type, dropout_rate = dropout_rate)
        train_fn, valid_fn = dnn_model.build_finetune_functions(
                    (train_set_x, train_set_y), (valid_set_x, valid_set_y))  #, batch_size=batch_size

    else:
        logger.critical('%s type NN model is not supported!' %(model_type))
        raise

    logger.info('fine-tuning the %s model' %(model_type))

    start_time = time.time()

    best_dnn_model = dnn_model
    best_validation_loss = sys.float_info.max
    previous_loss = sys.float_info.max

    early_stop = 0
    epoch = 0

#    finetune_lr = 0.000125
    previous_finetune_lr = finetune_lr

    print   finetune_lr

    while (epoch < training_epochs):
        epoch = epoch + 1

        current_momentum = momentum
        current_finetune_lr = finetune_lr
        if epoch <= warmup_epoch:
            current_finetune_lr = finetune_lr
            current_momentum = warmup_momentum
        else:
            current_finetune_lr = previous_finetune_lr * 0.5

        previous_finetune_lr = current_finetune_lr

        train_error = []
        sub_start_time = time.time()

        while (not train_data_reader.is_finish()):

            shared_train_set_xy, temp_train_set_x, temp_train_set_y = train_data_reader.load_one_partition()
#            train_set_x.set_value(numpy.asarray(temp_train_set_x, dtype=theano.config.floatX), borrow=True)
#            train_set_y.set_value(numpy.asarray(temp_train_set_y, dtype=theano.config.floatX), borrow=True)

            # if sequential training, the batch size will be the number of frames in an utterance
            if sequential_training == True:
                batch_size = temp_train_set_x.shape[0]

            n_train_batches = temp_train_set_x.shape[0] / batch_size
            for index in xrange(n_train_batches):
                ## send a batch to the shared variable, rather than pass the batch size and batch index to the finetune function
                train_set_x.set_value(numpy.asarray(temp_train_set_x[index*batch_size:(index + 1)*batch_size], dtype=theano.config.floatX), borrow=True)
                train_set_y.set_value(numpy.asarray(temp_train_set_y[index*batch_size:(index + 1)*batch_size], dtype=theano.config.floatX), borrow=True)

                this_train_error = train_fn(current_finetune_lr, current_momentum)

                train_error.append(this_train_error)

        train_data_reader.reset()

        logger.debug('calculating validation loss')
        validation_losses = []
        while (not valid_data_reader.is_finish()):
            shared_valid_set_xy, temp_valid_set_x, temp_valid_set_y = valid_data_reader.load_one_partition()
            valid_set_x.set_value(numpy.asarray(temp_valid_set_x, dtype=theano.config.floatX), borrow=True)
            valid_set_y.set_value(numpy.asarray(temp_valid_set_y, dtype=theano.config.floatX), borrow=True)

            this_valid_loss = valid_fn()

            validation_losses.append(this_valid_loss)
        valid_data_reader.reset()

        this_validation_loss = numpy.mean(validation_losses)

        this_train_valid_loss = numpy.mean(numpy.asarray(train_error))

        sub_end_time = time.time()

        loss_difference = this_validation_loss - previous_loss

        logger.info('epoch %i, validation error %f, train error %f  time spent %.2f' %(epoch, this_validation_loss, this_train_valid_loss, (sub_end_time - sub_start_time)))
        if plot:
            plotlogger.add_plot_point('training convergence','validation set',(epoch,this_validation_loss))
            plotlogger.add_plot_point('training convergence','training set',(epoch,this_train_valid_loss))
            plotlogger.save_plot('training convergence',title='Progress of training and validation error',xlabel='epochs',ylabel='error')

        if this_validation_loss < best_validation_loss:
            if epoch > 3:
                cPickle.dump(best_dnn_model, open(nnets_file_name, 'wb'))

            best_dnn_model = dnn_model
            best_validation_loss = this_validation_loss
#            logger.debug('validation loss decreased, so saving model')

        if this_validation_loss >= previous_loss:
            logger.debug('validation loss increased')

#            dbn = best_dnn_model
            early_stop += 1

        if epoch > 15 and early_stop > early_stop_epoch:
            logger.debug('stopping early')
            break

        if math.isnan(this_validation_loss):
            break

        previous_loss = this_validation_loss

    end_time = time.time()
#    cPickle.dump(best_dnn_model, open(nnets_file_name, 'wb'))

    logger.info('overall  training time: %.2fm validation error %f' % ((end_time - start_time) / 60., best_validation_loss))

    if plot:
        plotlogger.save_plot('training convergence',title='Final training and validation error',xlabel='epochs',ylabel='error')

    return  best_validation_loss


def dnn_generation(valid_file_list, nnets_file_name, n_ins, n_outs, out_file_list):
    logger = logging.getLogger("dnn_generation")
    logger.debug('Starting dnn_generation')

    plotlogger = logging.getLogger("plotting")

    dnn_model = cPickle.load(open(nnets_file_name, 'rb'))

    file_number = len(valid_file_list)

    for i in xrange(file_number):  #file_number
        logger.info('generating %4d of %4d: %s' % (i+1,file_number,valid_file_list[i]) )
        fid_lab = open(valid_file_list[i], 'rb')
        features = numpy.fromfile(fid_lab, dtype=numpy.float32)
        fid_lab.close()
        features = features[:(n_ins * (features.size / n_ins))]
        test_set_x = features.reshape((-1, n_ins))

        predicted_parameter = dnn_model.parameter_prediction(test_set_x)

        ### write to cmp file
        predicted_parameter = numpy.array(predicted_parameter, 'float32')
        temp_parameter = predicted_parameter
        fid = open(out_file_list[i], 'wb')
        predicted_parameter.tofile(fid)
        logger.debug('saved to %s' % out_file_list[i])
        fid.close()

def dnn_generation_lstm(valid_file_list, nnets_file_name, n_ins, n_outs, out_file_list):
    logger = logging.getLogger("dnn_generation")
    logger.debug('Starting dnn_generation')

    plotlogger = logging.getLogger("plotting")

    dnn_model = cPickle.load(open(nnets_file_name, 'rb'))

    visualize_dnn(dnn_model)

    file_number = len(valid_file_list)

    for i in xrange(file_number):  #file_number
        logger.info('generating %4d of %4d: %s' % (i+1,file_number,valid_file_list[i]) )
        fid_lab = open(valid_file_list[i], 'rb')
        features = numpy.fromfile(fid_lab, dtype=numpy.float32)
        fid_lab.close()
        features = features[:(n_ins * (features.size / n_ins))]
        test_set_x = features.reshape((-1, n_ins))

        predicted_parameter = dnn_model.parameter_prediction_lstm(test_set_x)

        ### write to cmp file
        predicted_parameter = numpy.array(predicted_parameter, 'float32')
        temp_parameter = predicted_parameter
        fid = open(out_file_list[i], 'wb')
        predicted_parameter.tofile(fid)
        logger.debug('saved to %s' % out_file_list[i])
        fid.close()

##generate bottleneck layer as festures
def dnn_hidden_generation(valid_file_list, nnets_file_name, n_ins, n_outs, out_file_list):
    logger = logging.getLogger("dnn_generation")
    logger.debug('Starting dnn_generation')

    plotlogger = logging.getLogger("plotting")

    dnn_model = cPickle.load(open(nnets_file_name, 'rb'))

    file_number = len(valid_file_list)

    for i in xrange(file_number):
        logger.info('generating %4d of %4d: %s' % (i+1,file_number,valid_file_list[i]) )
        fid_lab = open(valid_file_list[i], 'rb')
        features = numpy.fromfile(fid_lab, dtype=numpy.float32)
        fid_lab.close()
        features = features[:(n_ins * (features.size / n_ins))]
        features = features.reshape((-1, n_ins))
        temp_set_x = features.tolist()
        test_set_x = theano.shared(numpy.asarray(temp_set_x, dtype=theano.config.floatX))

        predicted_parameter = dnn_model.generate_top_hidden_layer(test_set_x=test_set_x)

        ### write to cmp file
        predicted_parameter = numpy.array(predicted_parameter, 'float32')
        temp_parameter = predicted_parameter
        fid = open(out_file_list[i], 'wb')
        predicted_parameter.tofile(fid)
        logger.debug('saved to %s' % out_file_list[i])
        fid.close()

def make_equal(in_file_list, ref_file_list, in_feature_dim, ref_feature_dim):
    io_funcs = BinaryIOCollection()
    utt_number = len(in_file_list)

    for i in xrange(utt_number):
        in_file_name = in_file_list[i]
        in_features, in_frame_number = io_funcs.load_binary_file_frame(in_file_name, in_feature_dim)

        ref_file_name = ref_file_list[i]
        ref_features, ref_frame_number = io_funcs.load_binary_file_frame(ref_file_name, ref_feature_dim)

        print in_file_name, in_frame_number, ref_file_name, ref_frame_number

        target_features = numpy.zeros((ref_frame_number, in_feature_dim))
        if in_frame_number == ref_frame_number:
            continue;
        elif in_frame_number > ref_frame_number:
            target_features[0:ref_frame_number, ] = in_features[0:ref_frame_number, ]
            print ref_frame_number, in_file_name
            io_funcs.array_to_binary_file(target_features, in_file_name)
        elif in_frame_number < ref_frame_number:
            target_features[0:in_frame_number, ] = ref_features[0:in_frame_number, ]
            print in_frame_number, ref_file_name
            io_funcs.array_to_binary_file(target_features, ref_file_name)

def do_norm_lab(cfg, in_label_align_file_list, nn_label_file_list, label_norm_file, nn_label_norm_file_list, dur_file_list):
    logger.info('preparing label data (input) using cppmary style labels')
    label_cppmary = LabelCppmary()
    if cfg.add_frame_features:
        label_cppmary.prepare_acoustic_label_feature(in_label_align_file_list, nn_label_file_list)
    else:
        label_cppmary.prepare_label_feature(in_label_align_file_list, nn_label_file_list)

    assert (cfg.lab_dim == label_cppmary.label_dimension)

    min_max_normaliser = MinMaxNormalisation(feature_dimension=cfg.lab_dim, min_value=0.01, max_value=0.99)
    ###use only training data to find min-max information, then apply on the whole dataset
    if cfg.GenTestList:
        min_max_normaliser.load_min_max_values(label_norm_file)
    else:
        min_max_normaliser.find_min_max_values(nn_label_file_list[0:cfg.train_file_number])
    ### enforce silence such that the normalization runs without removing silence: only for final synthesis
    # if cfg.GenTestList and cfg.enforce_silence:
    #     min_max_normaliser.normalise_data(binary_label_file_list, nn_label_norm_file_list)
    # else:
    min_max_normaliser.normalise_data(nn_label_file_list, nn_label_norm_file_list)

    if min_max_normaliser != None and not cfg.GenTestList:
        ### save label normalisation information for unseen testing labels
        label_min_vector = min_max_normaliser.min_vector
        label_max_vector = min_max_normaliser.max_vector
        label_norm_info = numpy.concatenate((label_min_vector, label_max_vector), axis=0)

        label_norm_info = numpy.array(label_norm_info, 'float32')
        fid = open(label_norm_file, 'wb')
        label_norm_info.tofile(fid)
        fid.close()
        logger.info('saved %s vectors to %s' % (label_min_vector.size, label_norm_file))



def make_cmp(delta_win, acc_win, in_file_list_dict, nn_cmp_file_list, dur_file_list, lf0_file_list):
    logger.info('creating acoustic (output) features')
    acoustic_worker = AcousticComposition(delta_win=delta_win, acc_win=acc_win)
    if 'dur' in cfg.in_dir_dict.keys() and cfg.AcousticModel:
        acoustic_worker.make_equal_frames(dur_file_list, lf0_file_list, cfg.in_dimension_dict)
    acoustic_worker.prepare_nn_data(in_file_list_dict, nn_cmp_file_list, cfg.in_dimension_dict, cfg.out_dimension_dict)

def norm_cmp(cfg, nn_cmp_file_list, nn_cmp_norm_file_list, norm_info_file, var_dir, var_file_dict):
    logger.info('normalising acoustic (output) features using method %s' % cfg.output_feature_normalisation)
    cmp_norm_info = None
    if cfg.output_feature_normalisation == 'MVN':
        normaliser = MeanVarianceNorm(feature_dimension=cfg.cmp_dim)
        normaliser.compute_global_variance(nn_cmp_file_list[0:cfg.train_file_number], cfg.cmp_dim, var_dir)
        ###calculate mean and std vectors on the training data, and apply on the whole dataset
        global_mean_vector = normaliser.compute_mean(nn_cmp_file_list[0:cfg.train_file_number], 0, cfg.cmp_dim)
        global_std_vector = normaliser.compute_std(nn_cmp_file_list[0:cfg.train_file_number], global_mean_vector, 0,
                                                   cfg.cmp_dim)

        normaliser.feature_normalisation(nn_cmp_file_list, nn_cmp_norm_file_list)
        cmp_norm_info = numpy.concatenate((global_mean_vector, global_std_vector), axis=0)

    elif cfg.output_feature_normalisation == 'MINMAX':
        min_max_normaliser = MinMaxNormalisation(feature_dimension=cfg.cmp_dim)
        global_mean_vector = min_max_normaliser.compute_mean(nn_cmp_file_list[0:cfg.train_file_number])
        global_std_vector = min_max_normaliser.compute_std(nn_cmp_file_list[0:cfg.train_file_number],
                                                           global_mean_vector)

        min_max_normaliser = MinMaxNormalisation(feature_dimension=cfg.cmp_dim, min_value=0.01, max_value=0.99)
        min_max_normaliser.find_min_max_values(nn_cmp_file_list[0:cfg.train_file_number])
        min_max_normaliser.normalise_data(nn_cmp_file_list, nn_cmp_norm_file_list)

        cmp_min_vector = min_max_normaliser.min_vector
        cmp_max_vector = min_max_normaliser.max_vector
        cmp_norm_info = numpy.concatenate((cmp_min_vector, cmp_max_vector), axis=0)

    else:
        logger.critical('Normalisation type %s is not supported!\n' % (cfg.output_feature_normalisation))
        raise

    cmp_norm_info = numpy.array(cmp_norm_info, 'float32')
    fid = open(norm_info_file, 'wb')
    cmp_norm_info.tofile(fid)
    fid.close()
    logger.info('saved %s vectors to %s' % (cfg.output_feature_normalisation, norm_info_file))

    feature_index = 0
    for feature_name in cfg.out_dimension_dict.keys():
        feature_std_vector = numpy.array(
            global_std_vector[:, feature_index:feature_index + cfg.out_dimension_dict[feature_name]], 'float32')

        fid = open(var_file_dict[feature_name], 'w')
        feature_var_vector = feature_std_vector ** 2
        feature_var_vector.tofile(fid)
        fid.close()

        logger.info('saved %s variance vector to %s' % (feature_name, var_file_dict[feature_name]))

        feature_index += cfg.out_dimension_dict[feature_name]
    total_var_file = os.path.join(var_dir, 'total_var')
    fid = open(total_var_file, 'w')
    total_var = numpy.array(global_std_vector[:, :], 'float32')
    total_var = total_var ** 2
    total_var.tofile(fid)
    logger.info('saved total variance vector to %s' % (total_var_file))
    fid.close()

def merge_dnn(prefix, label_norm_file, lab_dim, norm_info_file):
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, 0)
    first_layer_w = arg_params['fc1_weight'].asnumpy()
    first_layer_b = arg_params['fc1_bias'].asnumpy()

    print first_layer_w.shape, first_layer_b.shape

    io_fun = BinaryIOCollection()

    lab_min_max, dims = io_fun.load_binary_file_frame(label_norm_file, lab_dim)
    lab_min_vec = lab_min_max[0, :]
    lab_max_vec = lab_min_max[1, :]
    target_min = 0.01
    target_max = 0.99

    print lab_min_vec.shape, lab_max_vec.shape

    lab_diff_vec = lab_max_vec - lab_min_vec

    lab_diff_vec[lab_diff_vec == 0] = 1.0

    target_diff = target_max - target_min

    factor_w = first_layer_w.copy()
    b = first_layer_b.copy()

    print 'orig w mean'
    print np.mean(first_layer_w)

    for i in xrange(lab_dim):
        factor_w[:, i] = factor_w[:, i] / lab_diff_vec[i]

    print np.mean(factor_w)

    w = factor_w * target_diff

    b = b + factor_w.dot(target_min * lab_diff_vec - lab_min_vec * target_diff)

    print first_layer_b[1:10]
    print b[1:10]
    print np.mean(w)
    print np.mean(factor_w)

    first_layer_w = w.copy()
    first_layer_b = b.copy()

    print 'process the cmp file'
    final_layer_w = arg_params['fc7_weight'].asnumpy()
    final_layer_b = arg_params['fc7_bias'].asnumpy()

    cmp_mean_var, dims = io_fun.load_binary_file_frame(norm_info_file, cfg.cmp_dim)
    cmp_mean = cmp_mean_var[0, :]
    cmp_var = cmp_mean_var[1, :]
    print cmp_mean.shape, cmp_var.shape

    w = final_layer_w.copy()
    b = final_layer_b.copy()

    for i in xrange(cfg.cmp_dim):
        w[i, :] = w[i, :] * cmp_var[i]

    b = b * cmp_var + cmp_mean

    # print final_layer_b
    # print b
    # print final_layer_w[:, 0]
    # print cmp_var
    # print w[:, 0]

    final_layer_w = w.copy()
    final_layer_b = b.copy()

    arg_params['fc1_weight'] = mx.nd.array(first_layer_w)
    arg_params['fc1_bias'] = mx.nd.array(first_layer_b)
    arg_params['fc7_weight'] = mx.nd.array(final_layer_w)
    arg_params['fc7_bias'] = mx.nd.array(final_layer_b)
    mx.model.save_checkpoint(prefix, 100, sym, arg_params, aux_params)

def do_train(cfg, var_file_dict, norm_info_file, model_dir, nnets_file_name, train_x_file_list, train_y_file_list, valid_x_file_list, valid_y_file_list):
    hidden_dim = cfg.hidden_dim
    n_epoch = cfg.training_epochs
    model_prefix = cfg.model_prefix
    var_dict = load_covariance(var_file_dict, cfg.out_dimension_dict)
    logger.info('training DNN')
    fid = open(norm_info_file, 'rb')
    cmp_min_max = numpy.fromfile(fid, dtype=numpy.float32)
    fid.close()
    cmp_min_max = cmp_min_max.reshape((2, -1))
    cmp_mean_vector = cmp_min_max[0,]
    cmp_std_vector = cmp_min_max[1,]

    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # not an error - just means directory already exists
            pass
        else:
            logger.critical('Failed to create model directory %s' % model_dir)
            logger.critical(' OS error was: %s' % e.strerror)
            raise

    try:
        if cfg.framework == 'mxnet':
            batch_size = int(cfg.hyper_params['batch_size'])
            sequential_training = False
            n_ins = cfg.lab_dim
            n_outs = cfg.cmp_dim
            input_dim = n_ins
            output_dim = n_outs
            model_dnn = MxnetTTs(input_dim, output_dim, hidden_dim, batch_size, n_epoch, model_prefix)
            train_dataiter_all = TTSIter(x_file_list=train_x_file_list, y_file_list=train_y_file_list, n_ins=n_ins,
                                         n_outs=n_outs, batch_size=batch_size, sequential=sequential_training, shuffle=True)
            val_dataiter_all = TTSIter(x_file_list=valid_x_file_list, y_file_list=valid_y_file_list, n_ins=n_ins,
                                       n_outs=n_outs, batch_size=batch_size, sequential=sequential_training, shuffle=False)
            # model_dnn.train(train_dataiter, val_dataiter)

            # train_x_file_list1 = train_x_file_list[0:len(train_x_file_list)/2]
            # train_x_file_list2 = train_x_file_list[len(train_x_file_list)/2:]
            # train_y_file_list1 = train_y_file_list[0:len(train_y_file_list)/2]
            # train_y_file_list2 = train_y_file_list[len(train_y_file_list)/2:]
            # valid_x_file_list1 = valid_x_file_list[0:len(valid_x_file_list)/2]
            # valid_x_file_list2 = valid_x_file_list[len(valid_x_file_list)/2:]
            # valid_y_file_list1 = valid_y_file_list[0:len(valid_y_file_list)/2]
            # valid_y_file_list2 = valid_y_file_list[len(valid_y_file_list)/2:]
            # train_dataiter1 = TTSIter(x_file_list = train_x_file_list1, y_file_list = train_y_file_list1, n_ins = n_ins, n_outs = n_outs, batch_size = batch_size, sequential = sequential_training, shuffle = True)
            # train_dataiter2 = TTSIter(x_file_list = train_x_file_list2, y_file_list = train_y_file_list2, n_ins = n_ins, n_outs = n_outs, batch_size = batch_size, sequential = sequential_training, shuffle = True)
            # val_dataiter1 = TTSIter(x_file_list = valid_x_file_list1, y_file_list = valid_y_file_list1, n_ins = n_ins, n_outs = n_outs, batch_size = batch_size, sequential = sequential_training, shuffle = False)
            # val_dataiter2 = TTSIter(x_file_list = valid_x_file_list2, y_file_list = valid_y_file_list2, n_ins = n_ins, n_outs = n_outs, batch_size = batch_size, sequential = sequential_training, shuffle = False)

            # train_dataiter = mx.io.PrefetchingIter([train_dataiter1, train_dataiter2], rename_data=[{'data': 'data1'}, {'data': 'data2'}])
            # val_dataiter = mx.io.PrefetchingIter([val_dataiter1, val_dataiter2], rename_data=[{'data': 'data1'}, {'data': 'data2'}])
            # train_dataiter = mx.io.PrefetchingIter([train_dataiter1, train_dataiter2], rename_data = [{'data': 'data1'}, {'data': 'data2'}], rename_label = [{'label': 'label1'}, {'label': 'label2'}])
            # val_dataiter = mx.io.PrefetchingIter([val_dataiter1, val_dataiter2], rename_data = [{'data': 'data1'}, {'data': 'data2'}], rename_label = [{'label': 'label1'}, {'label': 'label2'}])
            # train_dataiter = mx.io.PrefetchingIter([train_dataiter1, train_dataiter2])
            # val_dataiter = mx.io.PrefetchingIter([val_dataiter1, val_dataiter2])
            train_dataiter = mx.io.PrefetchingIter(train_dataiter_all)
            val_dataiter = mx.io.PrefetchingIter(val_dataiter_all)
            # model_dnn.train(train_dataiter, val_dataiter)
            model_dnn.train_module(train_dataiter, val_dataiter)
            print "model train ok"

        else:
            train_DNN(train_xy_file_list=(train_x_file_list, train_y_file_list), \
                      valid_xy_file_list=(valid_x_file_list, valid_y_file_list), \
                      nnets_file_name=nnets_file_name, \
                      n_ins=cfg.lab_dim, n_outs=cfg.cmp_dim, ms_outs=cfg.multistream_outs, \
                      hyper_params=cfg.hyper_params, buffer_size=cfg.buffer_size, plot=cfg.plot, var_dict=var_dict,
                      cmp_mean_vector=cmp_mean_vector, cmp_std_vector=cmp_std_vector)
    except KeyboardInterrupt:
        logger.critical('train_DNN interrupted via keyboard')
        # Could 'raise' the exception further, but that causes a deep traceback to be printed
        # which we don't care about for a keyboard interrupt. So, just bail out immediately
        sys.exit(1)
    except:
        logger.critical('train_DNN threw an exception')
        raise


def do_generate(cfg, gen_dir, gen_file_id_list, test_x_file_list, nnets_file_name, norm_info_file, var_file_dict):
    logger.info('generating from DNN')
    try:
        os.makedirs(gen_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # not an error - just means directory already exists
            pass
        else:
            logger.critical('Failed to create generation directory %s' % gen_dir)
            logger.critical(' OS error was: %s' % e.strerror)
            raise

    gen_file_list = prepare_file_path_list(gen_file_id_list, gen_dir, cfg.cmp_ext)

    if cfg.framework == 'mxnet':
        prefix = '%s-%04d-%03d' % (cfg.model_prefix, cfg.hidden_dim, cfg.training_epochs)
        print prefix
        model_dnn = mx.model.FeedForward.load(prefix, 0)
        dnn_generation_mxnet(test_x_file_list, model_dnn, cfg.lab_dim, cfg.cmp_dim, gen_file_list)
    else:
        dnn_generation(test_x_file_list, nnets_file_name, cfg.lab_dim, cfg.cmp_dim, gen_file_list)

        logger.debug('denormalising generated output using method %s' % cfg.output_feature_normalisation)

    fid = open(norm_info_file, 'rb')
    cmp_min_max = numpy.fromfile(fid, dtype=numpy.float32)
    fid.close()
    cmp_min_max = cmp_min_max.reshape((2, -1))
    cmp_min_vector = cmp_min_max[0,]
    cmp_max_vector = cmp_min_max[1,]

    if cfg.output_feature_normalisation == 'MVN':
        denormaliser = MeanVarianceNorm(feature_dimension=cfg.cmp_dim)
        denormaliser.feature_denormalisation(gen_file_list, gen_file_list, cmp_min_vector, cmp_max_vector)

    elif cfg.output_feature_normalisation == 'MINMAX':
        denormaliser = MinMaxNormalisation(cfg.cmp_dim, min_value=0.01, max_value=0.99, min_vector=cmp_min_vector,
                                           max_vector=cmp_max_vector)
        denormaliser.denormalise_data(gen_file_list, gen_file_list)
    else:
        logger.critical('denormalising method %s is not supported!\n' % (cfg.output_feature_normalisation))
        raise

    if cfg.AcousticModel:
        ##perform MLPG to smooth parameter trajectory
        ## lf0 is included, the output features much have vuv.
        generator = ParameterGeneration(gen_wav_features=cfg.gen_wav_features, enforce_silence=cfg.enforce_silence)
        generator.acoustic_decomposition(gen_file_list, cfg.cmp_dim, cfg.out_dimension_dict, cfg.file_extension_dict,
                                         var_file_dict, do_MLPG=cfg.do_MLPG, cfg=cfg)

    if cfg.DurationModel:
        ### Perform duration normalization(min. state dur set to 1) ###
        gen_dur_list = prepare_file_path_list(gen_file_id_list, gen_dir, cfg.dur_ext)
        gen_label_list = prepare_file_path_list(gen_file_id_list, gen_dir, cfg.lab_ext)
        in_gen_label_align_file_list = prepare_file_path_list(gen_file_id_list, cfg.in_label_align_dir, cfg.lab_ext,
                                                              False)

        generator = ParameterGeneration(gen_wav_features=cfg.gen_wav_features)
        generator.duration_decomposition(gen_file_list, cfg.cmp_dim, cfg.out_dimension_dict, cfg.file_extension_dict)

        label_cppmary1 = LabelCppmary()
        label_cppmary1.prepare_predict_label(in_gen_label_align_file_list, gen_label_list, gen_dur_list)

    ### generate wav
    if cfg.GENWAV:
        logger.info('reconstructing waveform(s)')
        generate_wav(gen_dir, gen_file_id_list, cfg)  # generated speech

def main_function(cfg):

    logger = logging.getLogger("main")
    plotlogger = logging.getLogger("plotting")
    plotlogger.set_plot_path(cfg.plot_dir)
    hidden_layer_size = cfg.hyper_params['hidden_layer_size']


    ####prepare environment
    try:
        file_id_list = read_file_list(cfg.file_id_scp)
        random.seed(281638)
        random.shuffle(file_id_list)
        total_num = len(file_id_list)
        if cfg.train_file_number < 0 :
            cfg.train_file_number = int(total_num * 0.9)
            cfg.valid_file_number = int(total_num * 0.1)
            logger.debug('###### train valid patition: %d : %d' % (cfg.train_file_number, cfg.valid_file_number))
        logger.debug('Loaded file id list from %s' % cfg.file_id_scp)
    except IOError:
        # this means that open(...) threw an error
        logger.critical('Could not load file id list from %s' % cfg.file_id_scp)
        raise


    ###total file number including training, development, and testing
    total_file_number = len(file_id_list)

    data_dir = cfg.data_dir

    nn_cmp_dir       = os.path.join(data_dir, 'nn' + cfg.combined_feature_name + '_' + str(cfg.cmp_dim))
    nn_cmp_norm_dir   = os.path.join(data_dir, 'nn_norm'  + cfg.combined_feature_name + '_' + str(cfg.cmp_dim))

    model_dir = os.path.join(cfg.work_dir, 'nnets_model')
    gen_dir   = os.path.join(cfg.work_dir, 'gen')

    in_file_list_dict = {}

    for feature_name in cfg.in_dir_dict.keys():
        in_file_list_dict[feature_name] = prepare_file_path_list(file_id_list, cfg.in_dir_dict[feature_name], cfg.file_extension_dict[feature_name], False)

    nn_cmp_file_list         = prepare_file_path_list(file_id_list, nn_cmp_dir, cfg.cmp_ext)
    nn_cmp_norm_file_list    = prepare_file_path_list(file_id_list, nn_cmp_norm_dir, cfg.cmp_ext)

    norm_info_file = os.path.join(data_dir, 'norm_info' + cfg.combined_feature_name + '_' + str(cfg.cmp_dim) + '_' + cfg.output_feature_normalisation + '.dat')

    lab_dim = cfg.lab_dim
    suffix = str(lab_dim)
    #suffix = "cppmary"

    if cfg.process_labels_in_work_dir:
        label_data_dir = cfg.work_dir
    else:
        label_data_dir = data_dir

    # the number can be removed
    binary_label_dir      = os.path.join(label_data_dir, 'binary_label_'+suffix)
    nn_label_dir          = os.path.join(label_data_dir, 'nn_no_silence_lab_'+suffix)
    nn_label_norm_dir     = os.path.join(label_data_dir, 'nn_no_silence_lab_norm_'+suffix)

    in_label_align_file_list = prepare_file_path_list(file_id_list, cfg.in_label_align_dir, cfg.lab_ext, False)
    nn_label_file_list       = prepare_file_path_list(file_id_list, nn_label_dir, cfg.lab_ext)
    nn_label_norm_file_list  = prepare_file_path_list(file_id_list, nn_label_norm_dir, cfg.lab_ext)
    dur_file_list            = prepare_file_path_list(file_id_list, cfg.in_dur_dir, cfg.dur_ext)
    lf0_file_list            = prepare_file_path_list(file_id_list, cfg.in_lf0_dir, cfg.lf0_ext)

    # to do - sanity check the label dimension here?

    min_max_normaliser = None
    label_norm_file = 'label_norm_%s_%d.dat' %(cfg.label_style, lab_dim)
    label_norm_file = os.path.join(label_data_dir, label_norm_file)

    if cfg.GenTestList:
        try:
            test_id_list = read_file_list(cfg.test_id_scp)
            logger.debug('Loaded file id list from %s' % cfg.test_id_scp)
        except IOError:
            # this means that open(...) threw an error
            logger.critical('Could not load file id list from %s' % cfg.test_id_scp)
            raise

        in_label_align_file_list = prepare_file_path_list(test_id_list, cfg.in_label_align_dir, cfg.lab_ext, False)
        nn_label_file_list       = prepare_file_path_list(test_id_list, nn_label_dir, cfg.lab_ext)
        nn_label_norm_file_list  = prepare_file_path_list(test_id_list, nn_label_norm_dir, cfg.lab_ext)

    ### save acoustic normalisation information for normalising the features back
    var_dir   = os.path.join(data_dir, 'var')
    if not os.path.exists(var_dir):
        os.makedirs(var_dir)

    var_file_dict = {}
    for feature_name in cfg.out_dimension_dict.keys():
        var_file_dict[feature_name] = os.path.join(var_dir, feature_name + '_' + str(cfg.out_dimension_dict[feature_name]))


    if cfg.NORMLAB and (cfg.label_style == 'cppmary'):
        do_norm_lab(cfg, in_label_align_file_list, nn_label_file_list, label_norm_file, nn_label_norm_file_list, dur_file_list)

    ### make output duration data
    if cfg.MAKEDUR:
        if cfg.label_style == 'cppmary':
            logger.info('creating cppmary duration (output) features')
            label_cppmary = LabelCppmary()
            label_cppmary.prepare_dur_feature(in_label_align_file_list, dur_file_list)
        else:
            logger.info('creating duration (output) features')
            raise

    ### make output acoustic data
    if cfg.MAKECMP: #如果有多个数据流则合并，并计算delta，delta-delta
        make_cmp(cfg.delta_win, cfg.acc_win, in_file_list_dict, nn_cmp_file_list, dur_file_list, lf0_file_list)


    ### normalise output acoustic data
    if cfg.NORMCMP:
        norm_cmp(cfg, nn_cmp_file_list, nn_cmp_norm_file_list, norm_info_file, var_dir, var_file_dict)


    train_x_file_list = nn_label_norm_file_list[0:cfg.train_file_number]
    train_y_file_list = nn_cmp_norm_file_list[0:cfg.train_file_number]
    valid_x_file_list = nn_label_norm_file_list[cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number]
    valid_y_file_list = nn_cmp_norm_file_list[cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number]
    gen_file_id_list = file_id_list[cfg.train_file_number+cfg.valid_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]
    test_x_file_list  = nn_label_norm_file_list[cfg.train_file_number+cfg.valid_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]

    logger.info('label dimension is %d' % lab_dim)

    combined_model_arch = str(len(hidden_layer_size))
    for hid_size in hidden_layer_size:
        combined_model_arch += '_' + str(hid_size)

    nnets_file_name = '%s/%s_%s_%d_%s_%d.%d.train.%d.%f.rnn.model' \
                      %(model_dir, cfg.combined_model_name, cfg.combined_feature_name, int(cfg.multistream_switch),
                        combined_model_arch, lab_dim, cfg.cmp_dim, cfg.train_file_number, cfg.hyper_params['learning_rate'])

    ### DNN model training
    if cfg.TRAINDNN:
        do_train(cfg, var_file_dict, norm_info_file, model_dir, nnets_file_name, train_x_file_list, train_y_file_list,
                 valid_x_file_list, valid_y_file_list)


    if cfg.DNNGEN:
        ### generate parameters from DNN
        temp_dir_name = 'test_gen'
        gen_dir = os.path.join(gen_dir, temp_dir_name)

        if cfg.GenTestList:
            gen_file_id_list = test_id_list
            test_x_file_list = nn_label_norm_file_list
            ### comment the below line if you don't want the files in a separate folder
            gen_dir = cfg.test_synth_dir
        do_generate(cfg, gen_dir, gen_file_id_list, test_x_file_list, nnets_file_name, norm_info_file, var_file_dict)


# load the mxnet dnn layer and merge the first and final layer into the dnn
    merge_norm_dnn = True
    if merge_norm_dnn and cfg.framework == 'mxnet':
        prefix = '%s-%04d-%03d' % (cfg.model_prefix, cfg.hidden_dim, cfg.training_epochs)
        merge_dnn(prefix, label_norm_file, lab_dim, norm_info_file)



if __name__ == '__main__':

    # these things should be done even before trying to parse the command line

    # create a configuration instance
    # and get a short name for this instance
    cfg=configuration.cfg

    # set up logging to use our custom class
    logging.setLoggerClass(LoggerPlotter)

    # get a logger for this main function
    logger = logging.getLogger("main")


    if len(sys.argv) != 2:
        logger.critical('usage: run_merlin.sh [config file name]')
        sys.exit(1)

    config_file = sys.argv[1]

    config_file = os.path.abspath(config_file)
    cfg.configure(config_file)


    logger.info('Installation information:')
    logger.info('  Merlin directory: '+os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)))
    logger.info('  PATH:')
    env_PATHs = os.getenv('PATH')
    if env_PATHs:
        env_PATHs = env_PATHs.split(':')
        for p in env_PATHs:
            if len(p)>0: logger.info('      '+p)
    logger.info('  LD_LIBRARY_PATH:')
    env_LD_LIBRARY_PATHs = os.getenv('LD_LIBRARY_PATH')
    if env_LD_LIBRARY_PATHs:
        env_LD_LIBRARY_PATHs = env_LD_LIBRARY_PATHs.split(':')
        for p in env_LD_LIBRARY_PATHs:
            if len(p)>0: logger.info('      '+p)
    logger.info('  Python version: '+sys.version.replace('\n',''))
    logger.info('    PYTHONPATH:')
    env_PYTHONPATHs = os.getenv('PYTHONPATH')
    if env_PYTHONPATHs:
        env_PYTHONPATHs = env_PYTHONPATHs.split(':')
        for p in env_PYTHONPATHs:
            if len(p) > 0:
                logger.info('      ' + p)
    logger.info('  Numpy version: ' + numpy.version.version)
    logger.info('  Theano version: ' + theano.version.version)
    # logger.info('    THEANO_FLAGS: '+os.getenv('THEANO_FLAGS'))
    # logger.info('    device: '+theano.config.device)

    # Check for the presence of git
    ret = os.system('git status > /dev/null')
    if ret==0:
        logger.info('  Git is available in the working directory:')
        git_describe = subprocess.Popen(['git', 'describe', '--tags', '--always'], stdout=subprocess.PIPE).communicate()[0][:-1]
        logger.info('    Merlin version: '+git_describe)
        git_branch = subprocess.Popen(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], stdout=subprocess.PIPE).communicate()[0][:-1]
        logger.info('    branch: '+git_branch)
        git_diff = subprocess.Popen(['git', 'diff', '--name-status'], stdout=subprocess.PIPE).communicate()[0]
        git_diff = git_diff.replace('\t',' ').split('\n')
        logger.info('    diff to Merlin version:')
        for filediff in git_diff:
            if len(filediff)>0: logger.info('      '+filediff)
        logger.info('      (all diffs logged in '+os.path.basename(cfg.log_file)+'.gitdiff'+')')
        os.system('git diff > '+cfg.log_file+'.gitdiff')

    logger.info('Execution information:')
    logger.info('  HOSTNAME: '+socket.getfqdn())
    logger.info('  USER: '+os.getenv('USER'))
    logger.info('  PID: '+str(os.getpid()))
    PBS_JOBID = os.getenv('PBS_JOBID')
    if PBS_JOBID:
        logger.info('  PBS_JOBID: '+PBS_JOBID)


    if cfg.profile:
        logger.info('profiling is activated')
        import cProfile, pstats
        cProfile.run('main_function(cfg)', 'mainstats')

        # create a stream for the profiler to write to
        profiling_output = StringIO.StringIO()
        p = pstats.Stats('mainstats', stream=profiling_output)

        # print stats to that stream
        # here we just report the top 10 functions, sorted by total amount of time spent in each
        p.strip_dirs().sort_stats('tottime').print_stats(10)

        # print the result to the log
        logger.info('---Profiling result follows---\n%s' %  profiling_output.getvalue() )
        profiling_output.close()
        logger.info('---End of profiling result---')

    else:
        main_function(cfg)

#    if gnp._boardId is not None:
#        import gpu_lock
#        gpu_lock.free_lock(gnp._boardId)

    sys.exit(0)
