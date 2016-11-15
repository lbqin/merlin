#!/usr//python
#coding=utf-8
import mxnet as mx
import numpy as np
import logging
import random
from io_funcs.binary_io import  BinaryIOCollection
import sys
import os

class MxnetTTs():
    def __init__(self, input_dim, output_dim, hidden_dim, batch_size, n_epoch, output_type):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        logging.basicConfig(level=logging.DEBUG)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_epoch = n_epoch
        self.output_type = output_type
        self.batch_size = batch_size
        self.network = self.get_net(self.input_dim, self.output_dim, self.hidden_dim)
        print self.network.list_arguments()


    def get_net(self, input_dim, output_dim, hidden_dim):
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
        fc6 = mx.symbol.FullyConnected(act5, name='fc6', num_hidden=hidden_dim)
        act6 = mx.symbol.Activation(fc6, name='tanh6', act_type="tanh")
        #dropout = mx.sym.Dropout(data=act6, p=0.25)
        fc7 = mx.symbol.FullyConnected(act6, name='fc7', num_hidden=output_dim)
        linear = mx.symbol.LinearRegressionOutput(data=fc7, name="linear",label=label)
        #mx.viz.plot_network(linear).render()
        return linear

    def train(self, train_dataiter, val_dataiter):
        train_dataiter.reset()
        metric = mx.metric.create('mse')
        devs = mx.gpu()
        if self.output_type == 'duration':
            step = 10000
        else:
            step = 100000
        stop_factor_lr = 1e-6
        lr = mx.lr_scheduler.FactorScheduler(step=step, factor=.9, stop_factor_lr=stop_factor_lr)
        optimizer = mx.optimizer.SGD(
                learning_rate = 0.001,
                wd = 0.0005,
                momentum=0.9,
                clip_gradient = 5.0,
                lr_scheduler = lr)

        #optimizer = mx.optimizer.Adam(
        #        learning_rate = 0.002,
        #        wd = 0.0005,
        #        beta1 = 0.5,
        #        clip_gradient = 5.0,
        #        lr_scheduler = lr,
        #        )

        initializer = mx.init.Xavier(factor_type="in", magnitude=2.34)
        model = mx.model.FeedForward(ctx = devs,
             symbol = self.network,
             num_epoch = self.n_epoch,
             optimizer = optimizer,
             initializer = initializer)

        model.fit(X = train_dataiter, eval_data = val_dataiter, eval_metric = metric, batch_end_callback = mx.callback.Speedometer(self.batch_size, 256))
        model.save(self.output_type, 0)

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
                 sequential=False, output_type=None, shuffle=False):
        self.n_ins = n_ins
        self.n_outs = n_outs
        self.batch_size = batch_size
        self.buffer_size = 5000
        self.sequential = sequential
        self.output_type = output_type
        self.buffer_size = int(self.buffer_size / self.batch_size) * batch_size
        self.n_train_batchs = 0
        self.batch_index = 0

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
        self.remain_frame_number = 0
        self.end_reading = False
        logging.info('initialised')
        self._data = None
        self._label = None
        self._get_batch()
        # load all data in the same time
        #if output_type == 'duration':
        #    self.buffer_size = 600000
        #else:
        #    self.buffer_size = 4000000
        #self.train_x_all, self.train_y_all = self.load_one_partition()
        #logging.info('load all data %d' % self.train_x_all.shape[0])



    def __iter__(self):
        while (not self.is_finish()):
            data_value = mx.nd.empty((self.batch_size, self.n_ins))
            label_value = mx.nd.empty((self.batch_size, self.n_outs))
            batch_size = self.batch_size
            temp_train_set_x, temp_train_set_y = self.load_one_partition()
            n_train_batches = temp_train_set_x.shape[0] / batch_size
            print 'load data... %d', temp_train_set_x.shape[0]
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

    def _get_batch(self):
        if self.n_train_batchs == 0  or self.batch_index == self.n_train_batchs-1 :
            self.temp_train_set_x, self.temp_train_set_y = self.load_one_partition()
            self.n_train_batchs = self.temp_train_set_x.shape[0] / self.batch_size
            self.batch_index = 0
        prev_index = self.batch_index * self.batch_size
        back_index = (self.batch_index + 1) * self.batch_size
        self._data = {'data' : mx.nd.array(self.temp_train_set_x[prev_index:back_index])}
        self._label = {'label' : mx.nd.array(self.temp_train_set_y[prev_index:back_index])}
        self.batch_index += 1
        #print self.batch_index

    def iter_next(self):
        return not self.is_finish()

    def next(self):
        if self.iter_next():
            self._get_batch()
            data_batch = mx.io.DataBatch(data=self._data.values(),
                                   label=self._label.values(),
                                   pad=self.getpad(), index=self.getindex())
            return data_batch
        else:
            raise StopIteration

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in self._data.items()]

    @property
    def provide_label(self):
        return [(k, v.shape) for k, v in self._label.items()]

    #def __iter__(self):
    #    data_value = mx.nd.empty((self.batch_size, self.n_ins))
    #    label_value = mx.nd.empty((self.batch_size, self.n_outs))
    #    batch_size = self.batch_size
    #    n_train_batches = self.train_x_all.shape[0] / batch_size
    #    for index in xrange(n_train_batches):
    #        data_value[:] = self.train_x_all[index*batch_size : (index+1)*batch_size]
    #        label_value[:] = self.train_y_all[index*batch_size : (index+1)*batch_size]
    #        data_all = [data_value]
    #        label_all = [label_value]
    #        data_names = ['data']
    #        label_names = ['label']

    #        yield SimpleBatch(data_names, data_all, label_names, label_all)


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
                #base_file_name = self.x_files_list[self.file_index].split('/')[-1].split('.')[0]
                #logging.info("the number of frames in label and acoustic features are different: %d vs %d (%s)" % (lab_frame_number, out_frame_number, base_file_name))
                if lab_frame_number > out_frame_number:
                    frame_number = out_frame_number
            else:
                base_file_name = self.x_files_list[self.file_index].split('/')[-1].split('.')[0]
                logging.info(
                    "the number of frames in label and acoustic features are different: %d vs %d (%s)" % (
                    lab_frame_number, out_frame_number, base_file_name))
                self.file_index += 1
                continue

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


def prepare_duration_data(lab_dim, cmp_dim):

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

    nn_label_norm_dir = os.path.join(label_data_dir, 'nn_no_silence_lab_norm_' + str(lab_dim))
    nn_cmp_norm_dir = os.path.join(data_dir, 'nn_norm' + combined_feature_name + '_' + str(cmp_dim))

    nn_label_norm_file_list = prepare_file_path_list(file_id_list, nn_label_norm_dir, ".lab")
    nn_cmp_norm_file_list = prepare_file_path_list(file_id_list, nn_cmp_norm_dir, ".cmp")
    train_x_file_list = nn_label_norm_file_list[0:train_file_number]
    valid_x_file_list = nn_label_norm_file_list[train_file_number:train_file_number + valid_file_number]
    train_y_file_list = nn_cmp_norm_file_list[0:train_file_number]
    valid_y_file_list = nn_cmp_norm_file_list[train_file_number:train_file_number + valid_file_number]
    return train_x_file_list, valid_x_file_list, train_y_file_list, valid_y_file_list

def prepare_acoustic_data(lab_dim, cmp_dim):
    train_file_number = 1000
    valid_file_number = 66

    exp_dir = "/home/sooda/speech/merlin/egs/slt_arctic/s1/experiments/slt_arctic_full/"
    label_data_dir = exp_dir + "acoustic_model/data/"
    data_dir = exp_dir + "acoustic_model/data/"
    combined_feature_name = "_mgc_lf0_vuv_bap"
    file_id_scp = data_dir + "file_id_list_demo.scp"
    try:
        file_id_list = read_file_list(file_id_scp)
        logging.info('Loaded file id list from %s' % file_id_scp)
    except IOError:
        # this means that open(...) threw an error
        logging.info('Could not load file id list from %s' % file_id_scp)
        raise

    nn_label_norm_dir = os.path.join(label_data_dir, 'nn_no_silence_lab_norm_' + str(lab_dim))
    nn_cmp_norm_dir = os.path.join(data_dir, 'nn_norm' + combined_feature_name + '_' + str(cmp_dim))

    print file_id_list[0]
    nn_label_norm_file_list = prepare_file_path_list(file_id_list, nn_label_norm_dir, ".lab")
    print file_id_list[0],nn_label_norm_file_list[0]
    nn_cmp_norm_file_list = prepare_file_path_list(file_id_list, nn_cmp_norm_dir, ".cmp")
    print file_id_list[0], nn_cmp_norm_file_list[0]
    train_x_file_list = nn_label_norm_file_list[0:train_file_number]
    valid_x_file_list = nn_label_norm_file_list[train_file_number:train_file_number + valid_file_number]
    train_y_file_list = nn_cmp_norm_file_list[0:train_file_number]
    valid_y_file_list = nn_cmp_norm_file_list[train_file_number:train_file_number + valid_file_number]
    print train_y_file_list[0], valid_y_file_list[0]
    return train_x_file_list, valid_x_file_list, train_y_file_list, valid_y_file_list


def dnn_generation_mxnet(valid_file_list, dnn_model, n_ins, n_outs, out_file_list):
    logging.info('Starting dnn_generation')

    file_number = len(valid_file_list)
    for i in xrange(file_number):
        logging.info('generating %4d of %4d: %s' % (i+1,file_number,valid_file_list[i]) )
        fid_lab = open(valid_file_list[i], 'rb')
        features = np.fromfile(fid_lab, dtype=np.float32)
        fid_lab.close()
        features = features[:(n_ins * (features.size / n_ins))]
        test_set_x = features.reshape((-1, n_ins))
        predicted_parameter = dnn_model.predict(test_set_x)
        ### write to cmp file
        predicted_parameter = np.array(predicted_parameter, 'float32')
        print features.shape, predicted_parameter.shape
        temp_parameter = predicted_parameter
        fid = open(out_file_list[i], 'wb')
        predicted_parameter.tofile(fid)
        logging.info('saved to %s' % out_file_list[i])
        fid.close()

def test(val_dataiter, model_prefix, num_epochs):
    print "test..."
    model_test = mx.model.FeedForward.load(model_prefix, num_epochs)
    #preds,data,label = model_test.predict(val_dataiter, 10, return_data=True)
    preds = []
    data = np.array([])
    label = np.array([])
    for batch in val_dataiter:
        for j, x in enumerate(batch.data):
            x = x.asnumpy()
            if data.shape[0]:
                data = np.concatenate((data, x), axis=0)
            else:
                data = x
        for j, y in enumerate(batch.label):
            y = y.asnumpy()
            if label.shape[0]:
                label = np.concatenate((label, y), axis=0)
            else:
                label = y

    preds = model_test.predict(data)
    for i in xrange(preds.shape[0]):
        print preds[i, 0:5]
        print label[i, 0:5]
        print "........."


    #for feats in data:
    #    preds.append(model_test.predict(feats))

    #for i in xrange(len(preds)):
    #    #print preds[i][0:5]
    #    #print label[i][0:5]
    #    print preds[i].shape, label[i].shape
    #    print "------------"

def setting_duration(input_dim, output_dim, batch_size):
    n_ins = input_dim
    n_outs = output_dim
    sequential_training = False

    train_x_file_list, valid_x_file_list, train_y_file_list, valid_y_file_list = prepare_duration_data(input_dim, output_dim)

    train_dataiter = TTSIter(x_file_list = train_x_file_list, y_file_list = train_y_file_list,
                                n_ins = n_ins, n_outs = n_outs, batch_size = batch_size, sequential = sequential_training, shuffle = True)
    val_dataiter = TTSIter(x_file_list = valid_x_file_list, y_file_list = valid_y_file_list,
                                n_ins = n_ins, n_outs = n_outs, batch_size = batch_size, sequential = sequential_training, shuffle = False)
    return train_dataiter, val_dataiter

def setting_acoustic(input_dim, output_dim, batch_size):
    n_ins = input_dim
    n_outs = output_dim
    sequential_training = False

    train_x_file_list, valid_x_file_list, train_y_file_list, valid_y_file_list = prepare_acoustic_data(input_dim, output_dim)

    train_dataiter = TTSIter(x_file_list = train_x_file_list, y_file_list = train_y_file_list,
                                n_ins = n_ins, n_outs = n_outs, batch_size = batch_size, sequential = sequential_training, shuffle = True)
    val_dataiter = TTSIter(x_file_list = valid_x_file_list, y_file_list = valid_y_file_list,
                                n_ins = n_ins, n_outs = n_outs, batch_size = batch_size, sequential = sequential_training, shuffle = False)
    return train_dataiter, val_dataiter

def test_generation():
    #model_prefix = 'duration'
    #n_ins = 416
    #n_outs = 5
    model_prefix = 'acoustic'
    n_ins = 425
    n_outs = 187
    num_epoch = 25
    model_test = mx.model.FeedForward.load(model_prefix, 0)
    test_dir = "/home/sooda/speech/merlin/egs/slt_arctic/s1/experiments/slt_arctic_full/test_synthesis/"
    #lab_dir = test_dir + "prompt-lab/"
    lab_dir = test_dir + "gen-lab/"
    cmp_dir = test_dir + "duration_out/"
    file_id_scp = test_dir + "test_id_list.scp"
    file_id_list = read_file_list(file_id_scp)
    valid_file_list = prepare_file_path_list(file_id_list, lab_dir, ".lab")
    cmp_file_list = prepare_file_path_list(file_id_list, cmp_dir, ".cmp")
    dnn_generation_mxnet(valid_file_list, model_test, n_ins, n_outs, cmp_file_list)


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(level=logging.DEBUG)
    #test_generation()
    #exit()
    train_type = "acoustic"
    if train_type == 'duration':
        input_dim = 416
        output_dim = 5
        hidden_dim = 512
        batch_size = 64
        n_epoch = 25
        train_dataiter, val_dataiter = setting_duration(input_dim, output_dim, batch_size)
        output_type = 'duration'
        duration_dnn = MxnetTTs(input_dim, output_dim, hidden_dim, batch_size, n_epoch, output_type)
        duration_dnn.train(train_dataiter, val_dataiter)
    else:
        input_dim = 425
        output_dim = 187
        hidden_dim = 512
        n_epoch = 25
        batch_size = 256
        train_dataiter, val_dataiter = setting_acoustic(input_dim, output_dim, batch_size)
        output_type = 'acoustic'
        acoustic_dnn = MxnetTTs(input_dim, output_dim, hidden_dim, batch_size, n_epoch, output_type)
        acoustic_dnn.train(train_dataiter, val_dataiter)

