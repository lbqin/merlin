#!/usr//python
#coding=utf-8
import mxnet as mx
import numpy as np
import logging
import random
from io_funcs.binary_io import  BinaryIOCollection
import sys
import os

class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names
        self.pad = 0

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]


class TTSIter(mx.io.DataIter):
    def __init__(self, x_file_list, y_file_list, n_ins=0, n_outs=0, batch_size=100,
                 sequential=False, network_type=None, shuffle=False):
        self.n_ins = n_ins
        self.n_outs = n_outs
        self.batch_size = batch_size
        self.buffer_size = 2000000
        self.sequential = sequential
        self.network_type = network_type
        self.buffer_size = int(self.buffer_size / self.batch_size) * batch_size

        # remove potential empty lines and end of line signs
        try:
            assert len(x_file_list) > 0
        except AssertionError:
            logging.info('first list is empty')
            raise

        try:
            assert len(y_file_list) > 0
        except AssertionError:
            logging.info('second list is empty')
            raise

        try:
            assert len(x_file_list) == len(y_file_list)
        except AssertionError:
            logging.info('two lists are of differing lengths: %d versus %d', len(x_file_list), len(y_file_list))
            raise

        self.x_files_list = x_file_list
        self.y_files_list = y_file_list

        logging.info('first  list of items from ...%s to ...%s' % (
        self.x_files_list[0].rjust(20)[-20:], self.x_files_list[-1].rjust(20)[-20:]))
        logging.info('second list of items from ...%s to ...%s' % (
        self.y_files_list[0].rjust(20)[-20:], self.y_files_list[-1].rjust(20)[-20:]))

        if shuffle:
            random.seed(271638)
            random.shuffle(self.x_files_list)
            random.seed(271638)
            random.shuffle(self.y_files_list)

        self.file_index = 0
        self.list_size = len(self.x_files_list)

        self.remain_data_x = np.empty((0, self.n_ins))
        self.remain_data_y = np.empty((0, self.n_outs))
        self.provide_data = [('data', (batch_size, n_ins))]
        self.provide_label = [('label', (batch_size, n_outs))]
        self.remain_frame_number = 0
        self.end_reading = False
        logging.info('initialised')


    def __iter__(self):
        while (not self.is_finish()):
            data_value = mx.nd.empty((self.batch_size, self.n_ins))
            label_value = mx.nd.empty((self.batch_size, self.n_outs))
            batch_size = self.batch_size
            temp_train_set_x, temp_train_set_y = self.load_one_partition()
            n_train_batches = temp_train_set_x.shape[0] / batch_size
            for index in xrange(n_train_batches):
                # print data_value.shape, temp_train_set_x.shape
                data_value[:] = temp_train_set_x[index*batch_size : (index+1)*batch_size]
                label_value[:] = temp_train_set_y[index*batch_size : (index+1)*batch_size]
                # print data_value.shape, label_value.shape
                data_all = [data_value]
                label_all = [label_value]
                data_names = ['data']
                label_names = ['label']

                yield SimpleBatch(data_names, data_all, label_names, label_all)


    def reset(self):
        """When all the files in the file list have been used for DNN training, reset the data provider to start a new epoch.

        """
        self.file_index = 0
        self.end_reading = False

        self.remain_frame_number = 0

        logging.info('reset')

    def load_one_partition(self):
        if self.sequential == True:
            if not self.network_type:
                temp_set_x, temp_set_y = self.load_next_utterance()
            elif self.network_type == "RNN":
                temp_set_x, temp_set_y = self.load_next_utterance()
            elif self.network_type == "CTC":
                temp_set_x, temp_set_y = self.load_next_utterance_CTC()
            else:
                sys.exit(1)
        else:
            temp_set_x, temp_set_y = self.load_one_block()

        return temp_set_x, temp_set_y

    def load_next_utterance(self):
        """Load the data for one utterance. This function will be called when utterance-by-utterance loading is required (e.g., sequential training).

        """

        temp_set_x = np.empty((self.buffer_size, self.n_ins))
        temp_set_y = np.empty((self.buffer_size, self.n_outs))

        io_fun = BinaryIOCollection()

        in_features, lab_frame_number = io_fun.load_binary_file_frame(self.x_files_list[self.file_index], self.n_ins)
        out_features, out_frame_number = io_fun.load_binary_file_frame(self.y_files_list[self.file_index], self.n_outs)

        frame_number = lab_frame_number
        if abs(lab_frame_number - out_frame_number) < 5:  ## we allow small difference here. may not be correct, but sometimes, there is one/two frames difference
            if lab_frame_number > out_frame_number:
                frame_number = out_frame_number
        else:
            base_file_name = self.x_files_list[self.file_index].split('/')[-1].split('.')[0]
            logging.info("the number of frames in label and acoustic features are different: %d vs %d (%s)" % (
            lab_frame_number, out_frame_number, base_file_name))
            raise

        temp_set_y = out_features[0:frame_number, ]
        temp_set_x = in_features[0:frame_number, ]

        self.file_index += 1

        if self.file_index >= self.list_size:
            self.end_reading = True
            self.file_index = 0


        return temp_set_x, temp_set_y



    def load_next_utterance_CTC(self):

        temp_set_x = np.empty((self.buffer_size, self.n_ins))
        temp_set_y = np.empty(self.buffer_size)

        io_fun = BinaryIOCollection()

        in_features, lab_frame_number = io_fun.load_binary_file_frame(self.x_files_list[self.file_index], self.n_ins)
        out_features, out_frame_number = io_fun.load_binary_file_frame(self.y_files_list[self.file_index], self.n_outs)

        frame_number = lab_frame_number
        temp_set_x = in_features[0:frame_number, ]

        temp_set_y = np.array([self.n_outs])
        for il in np.argmax(out_features, axis=1):
            temp_set_y = np.concatenate((temp_set_y, [il, self.n_outs]), axis=0)

        self.file_index += 1

        if self.file_index >= self.list_size:
            self.end_reading = True
            self.file_index = 0

        return temp_set_x, temp_set_y

    def load_one_block(self):
        """Load one block data. The number of frames will be the buffer size set during intialisation.

        """

        logging.info('loading one block')

        temp_set_x = np.empty((self.buffer_size, self.n_ins))
        temp_set_y = np.empty((self.buffer_size, self.n_outs))
        current_index = 0

        ### first check whether there are remaining data from previous utterance
        if self.remain_frame_number > 0:
            temp_set_x[current_index:self.remain_frame_number, ] = self.remain_data_x
            temp_set_y[current_index:self.remain_frame_number, ] = self.remain_data_y
            current_index += self.remain_frame_number

            self.remain_frame_number = 0

        io_fun = BinaryIOCollection()
        while True:
            if current_index >= self.buffer_size:
                break
            if self.file_index >= self.list_size:
                self.end_reading = True
                self.file_index = 0
                break

            in_features, lab_frame_number = io_fun.load_binary_file_frame(self.x_files_list[self.file_index],
                                                                          self.n_ins)
            out_features, out_frame_number = io_fun.load_binary_file_frame(self.y_files_list[self.file_index],
                                                                           self.n_outs)

            frame_number = lab_frame_number
            if abs(lab_frame_number - out_frame_number) < 5:  ## we allow small difference here. may not be correct, but sometimes, there is one/two frames difference
                if lab_frame_number > out_frame_number:
                    frame_number = out_frame_number
            else:
                base_file_name = self.x_files_list[self.file_index].split('/')[-1].split('.')[0]
                logging.info(
                    "the number of frames in label and acoustic features are different: %d vs %d (%s)" % (
                    lab_frame_number, out_frame_number, base_file_name))
                raise

            out_features = out_features[0:frame_number, ]
            in_features = in_features[0:frame_number, ]

            if current_index + frame_number <= self.buffer_size:
                temp_set_x[current_index:current_index + frame_number, ] = in_features
                temp_set_y[current_index:current_index + frame_number, ] = out_features

                current_index = current_index + frame_number
            else:  ## if current utterance cannot be stored in the block, then leave the remaining part for the next block
                used_frame_number = self.buffer_size - current_index
                temp_set_x[current_index:self.buffer_size, ] = in_features[0:used_frame_number, ]
                temp_set_y[current_index:self.buffer_size, ] = out_features[0:used_frame_number, ]
                current_index = self.buffer_size

                self.remain_data_x = in_features[used_frame_number:frame_number, ]
                self.remain_data_y = out_features[used_frame_number:frame_number, ]
                self.remain_frame_number = frame_number - used_frame_number

            self.file_index += 1

        temp_set_x = temp_set_x[0:current_index, ]
        temp_set_y = temp_set_y[0:current_index, ]

        np.random.seed(271639)
        np.random.shuffle(temp_set_x)
        np.random.seed(271639)
        np.random.shuffle(temp_set_y)

        return temp_set_x, temp_set_y




    def getpad(self):
        return 0

    def is_finish(self):
        return self.end_reading


##util function
def extract_file_id_list(file_list):
    file_id_list = []
    for file_name in file_list:
        file_id = os.path.basename(os.path.splitext(file_name)[0])
        file_id_list.append(file_id)

    return  file_id_list

def read_file_list(file_name):

    file_lists = []
    fid = open(file_name)
    for line in fid.readlines():
        line = line.strip()
        if len(line) < 1:
            continue
        file_lists.append(line)
    fid.close()

    logging.info('Read file list from %s' % file_name)
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


def prepare_data():

    train_file_number = 1000
    valid_file_number = 66

    exp_dir = "/home/sooda/speech/merlin/egs/slt_arctic/s1/experiments/slt_arctic_full/"
    label_data_dir = exp_dir + "duration_model/data/"
    data_dir = exp_dir + "duration_model/data/"
    combined_feature_name = "_dur"
    file_id_scp = data_dir + "file_id_list_demo.scp"
    try:
        file_id_list = read_file_list(file_id_scp)
        logging.info('Loaded file id list from %s' % file_id_scp)
    except IOError:
        # this means that open(...) threw an error
        logging.info('Could not load file id list from %s' % file_id_scp)
        raise

    nn_label_norm_dir = os.path.join(label_data_dir, 'nn_no_silence_lab_norm_' + suffix)
    nn_cmp_norm_dir = os.path.join(data_dir, 'nn_norm' + combined_feature_name + '_' + str(cmp_dim))

    nn_label_norm_file_list = prepare_file_path_list(file_id_list, nn_label_norm_dir, ".lab")
    nn_cmp_norm_file_list = prepare_file_path_list(file_id_list, nn_cmp_norm_dir, ".cmp")
    train_x_file_list = nn_label_norm_file_list[0:train_file_number]
    valid_x_file_list = nn_label_norm_file_list[train_file_number:train_file_number + valid_file_number]
    train_y_file_list = nn_cmp_norm_file_list[0:train_file_number]
    valid_y_file_list = nn_cmp_norm_file_list[train_file_number:train_file_number + valid_file_number]
    return train_x_file_list, valid_x_file_list, train_y_file_list, valid_y_file_list

def get_net():
    input_dim = 416
    output_dim = 5
    hidden_dim = 512
    data = mx.symbol.Variable('data')
    label = mx.symbol.Variable('label')
    fc1 = mx.symbol.FullyConnected(data, name='fc1', num_hidden=hidden_dim)
    act1 = mx.symbol.Activation(fc1, name='tanh1', act_type="tanh")
    fc2 = mx.symbol.FullyConnected(act1, name='fc2', num_hidden=hidden_dim)
    act2 = mx.symbol.Activation(fc2, name='tanh2', act_type="tanh")
    fc3 = mx.symbol.FullyConnected(act2, name='fc3', num_hidden=hidden_dim)
    act3 = mx.symbol.Activation(fc3, name='tanh3', act_type="tanh")
    fc4 = mx.symbol.FullyConnected(act3, name='fc4', num_hidden=hidden_dim)
    act4 = mx.symbol.Activation(fc4, name='tanh4', act_type="tanh")
    fc5 = mx.symbol.FullyConnected(act4, name='fc5', num_hidden=hidden_dim)
    act5 = mx.symbol.Activation(fc5, name='tanh5', act_type="tanh")
    fc6 = mx.symbol.FullyConnected(act5, name='fc6', num_hidden=output_dim)
    #act6 = mx.symbol.Activation(fc6, name='tanh6', act_type="tanh")
    linear = mx.symbol.LinearRegressionOutput(data=fc6, name="linear",label=label)
    mx.viz.plot_network(linear).render()

    #linear = mx.symbol.SoftmaxOutput(fc3, name='softmax')
    return linear

def test(val_dataiter, model_prefix, num_epochs):
    print "test..."
    model_test = mx.model.FeedForward.load(model_prefix, num_epochs)
    preds,data,label = model_test.predict(val_dataiter, 10, return_data=True)

    for i in xrange(len(preds)):
        print preds[i]
        print label[i]
        print "------------"



if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    net = get_net()
    print net.list_arguments()

    batch_size = 64
    n_epoch = 25
    lab_dim = 416
    model_prefix = 'duration'
    suffix = str(lab_dim)
    cmp_dim = 5
    n_ins = lab_dim
    n_outs = cmp_dim
    sequential_training = False
    train_type = 2
    only_test = 0

    train_x_file_list, valid_x_file_list, train_y_file_list, valid_y_file_list = prepare_data()

    train_dataiter = TTSIter(x_file_list = train_x_file_list, y_file_list = train_y_file_list,
                                n_ins = n_ins, n_outs = n_outs, batch_size = batch_size, sequential = sequential_training, shuffle = True)

    val_dataiter = TTSIter(x_file_list = valid_x_file_list, y_file_list = valid_y_file_list,
                                n_ins = n_ins, n_outs = n_outs, batch_size = batch_size, sequential = sequential_training, shuffle = False)

    if only_test:
        test(val_dataiter, model_prefix, n_epoch)
        exit()
    logging.basicConfig(level=logging.DEBUG)
    train_dataiter.reset()
    metric = mx.metric.create('mse')
    if train_type == 1:
        mod = mx.mod.Module(net)
        mod.fit(train_dataiter, eval_data=val_dataiter, eval_metric=metric,
            optimizer_params={'learning_rate':0.01, 'momentum': 0.9}, num_epoch=n_epoch)
        #evaluate on validation set with a evaluation metric
        mod.score(val_dataiter, metric)
        for name, val in metric.get_name_value():
            print('%s=%f' % (name, val))
    else:
        devs = mx.cpu()
        model = mx.model.FeedForward(ctx = devs,
                                         symbol = net,
                                         num_epoch = n_epoch,
                                         learning_rate = 0.002,
                                         wd = 0.0001,
                                         lr_scheduler=mx.lr_scheduler.FactorScheduler(2000,0.9),
                                         initializer = mx.init.Xavier(factor_type="in", magnitude=2.34), momentum = 0.9)

        model.fit(X = train_dataiter, eval_data = val_dataiter, eval_metric = metric, batch_end_callback = mx.callback.Speedometer(batch_size, 200))


        model.save(model_prefix, n_epoch)

        print "done"

