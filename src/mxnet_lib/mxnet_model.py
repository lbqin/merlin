#!/usr//python
#coding=utf-8
import mxnet as mx
import numpy as np
import logging
import random
import sys
import os
import time
import speechSGD


class SimpleLRScheduler(mx.lr_scheduler.LRScheduler):
    def __init__(self, dynamic_lr, momentum=0.3):
        super(SimpleLRScheduler, self).__init__()
        self.dynamic_lr = dynamic_lr
        self.momentum = momentum

    def __call__(self, num_update):
        return self.dynamic_lr, self.momentum


class MxnetTTs():
    def __init__(self, input_dim, output_dim, hidden_dim, batch_size, n_epoch, output_type, pretrain_name = ""):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        logging.basicConfig(level=logging.DEBUG)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_epoch = n_epoch
        self.output_type = output_type
        self.pretrain_name = pretrain_name
        self.batch_size = batch_size
        if output_type == 'phoneme':
            self.network = self.get_phoneme_net(self.input_dim, self.output_dim, self.hidden_dim)
        else:
            self.network = self.get_net(self.input_dim, self.output_dim, self.hidden_dim)
        print(self.network.list_arguments())


    def get_net(self, input_dim, output_dim, hidden_dim):
        data = mx.symbol.Variable('data')
        label = mx.symbol.Variable('label')
        #bn_mom = 0.9
        #data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
        net = mx.symbol.FullyConnected(data, name='fc1', num_hidden=hidden_dim)
        #net = mx.sym.BatchNorm(net, fix_gamma=True)
        net = mx.symbol.Activation(net, name='tanh1', act_type="tanh")
        net = mx.symbol.FullyConnected(net, name='fc2', num_hidden=hidden_dim)
        #net = mx.sym.BatchNorm(net, fix_gamma=True)
        net = mx.symbol.Activation(net, name='tanh2', act_type="tanh")
        net = mx.symbol.FullyConnected(net, name='fc3', num_hidden=hidden_dim)
        net = mx.symbol.Activation(net, name='tanh3', act_type="tanh")
        net = mx.symbol.FullyConnected(net, name='fc4', num_hidden=hidden_dim)
        net = mx.symbol.Activation(net, name='tanh4', act_type="tanh")
        net = mx.symbol.FullyConnected(net, name='fc5', num_hidden=hidden_dim)
        net = mx.symbol.Activation(net, name='tanh5', act_type="tanh")
        #net = mx.sym.Dropout(data=net, p=0.25)
        net = mx.symbol.FullyConnected(net, name='fc6', num_hidden=hidden_dim)
        net = mx.symbol.Activation(net, name='tanh6', act_type="tanh")
        #net = mx.sym.Dropout(data=net, p=0.25)
        net = mx.symbol.FullyConnected(net, name='fc7', num_hidden=output_dim)
        linear = mx.symbol.LinearRegressionOutput(data=net, name="linear",label=label)
        #mx.viz.plot_network(linear).render()
        return linear

    def get_phoneme_net(self, input_dim, output_dim, hidden_dim):
        data = mx.symbol.Variable('data')
        label = mx.symbol.Variable('label')
        #label = label.as_in_context(label.context).reshape((label.shape[0], ))
        bn_mom = 0.9
        data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
        net = mx.symbol.FullyConnected(data, name='fc1', num_hidden=hidden_dim)
        net = mx.sym.BatchNorm(net, fix_gamma=True)
        net = mx.symbol.Activation(net, name='tanh1', act_type="tanh")
        net = mx.symbol.FullyConnected(net, name='fc2', num_hidden=hidden_dim)
        net = mx.sym.BatchNorm(net, fix_gamma=True)
        net = mx.symbol.Activation(net, name='tanh2', act_type="tanh")
        net = mx.symbol.FullyConnected(net, name='fc3', num_hidden=hidden_dim)
        net = mx.symbol.Activation(net, name='tanh3', act_type="tanh")
        net = mx.symbol.FullyConnected(net, name='fc4', num_hidden=hidden_dim)
        net = mx.symbol.Activation(net, name='tanh4', act_type="tanh")
        net = mx.symbol.FullyConnected(net, name='fc5', num_hidden=hidden_dim)
        net = mx.symbol.Activation(net, name='tanh5', act_type="tanh")
        #net = mx.sym.Dropout(data=net, p=0.25)
        net = mx.symbol.FullyConnected(net, name='fc6', num_hidden=hidden_dim)
        net = mx.symbol.Activation(net, name='tanh6', act_type="tanh")
        #net = mx.sym.Dropout(data=net, p=0.25)
        net = mx.symbol.FullyConnected(net, name='fc7', num_hidden=output_dim)
        #linear = mx.symbol.LinearRegressionOutput(data=net, name="linear",label=label)
        sm = mx.symbol.SoftmaxOutput(data=net, name="softmax", label=label)
        #mx.viz.plot_network(linear).render()
        return sm

    def train_module(self, train_dataiter, val_dataiter):
        if self.output_type == 'duration':
            step = 10000
        else:
            step = 100000

        train_dataiter.reset()
        if self.output_type == 'phoneme':
            #metric = mx.metric.CrossEntropy()
            metric = mx.metric.create('acc')
        else:
            metric = mx.metric.create('mse')
        stop_factor_lr = 1e-6
        learning_rate = 0.001 #学习率太大，也会导致mse爆掉（达到几百）！
        clip_gradient = 5.0
        weight_decay = 0.0001
        momentum = 0.9
        lr_factor = 0.9
        warmup_momentum = 0.3
        devs = mx.gpu(0)
        # devs = mx.cpu()
        #lr = mx.lr_scheduler.FactorScheduler(step=step, factor=.9, stop_factor_lr=stop_factor_lr)
        lr = SimpleLRScheduler(learning_rate, momentum=warmup_momentum)
        initializer = mx.init.Xavier(factor_type="in", magnitude=2.34)

        mod = None
        batch_end_callbacks = [mx.callback.Speedometer(self.batch_size, self.batch_size * 4), ]

        self.prefix = '%s-%04d-%03d' % (self.output_type, self.hidden_dim, self.n_epoch)
        logging.info("save prefix: %s-%04d-%03d", self.output_type, self.hidden_dim, self.n_epoch)

        fixed_param_names = []
        use_pretrain = False
        if self.pretrain_name != "":
            use_pretrain = True
            use_fixed_param = False
            if use_fixed_param:
                # define layers with fixed weight/bias
                # fixed_param_names = [name for name in self.network.list_arguments()]
                fixed_param_names = ['fc1_weight', 'fc1_bias', 'fc2_weight', 'fc2_bias']
                logging.info("Freezed parameters: [" + ','.join(fixed_param_names) + ']')

        data_shapes = [('data', (self.batch_size, self.input_dim))]
        if self.output_type == 'phoneme':
            label_shapes = [('label', (self.batch_size,))]
        else:
            label_shapes = [('label', (self.batch_size, self.output_dim))]

        use_pretrain = False
        if self.pretrain_name != "":
            use_pretrain = True

        if use_pretrain:
            logging.info('loading checkpoint: %s', self.pretrain_name)
            sym, arg_params, aux_params = mx.model.load_checkpoint(self.pretrain_name, 0)
            mod = mx.mod.Module(sym, label_names=('label',), context=devs, fixed_param_names=fixed_param_names)
            mod.bind(data_shapes=data_shapes, label_shapes=label_shapes, for_training=True)
            mod.set_params(arg_params=arg_params, aux_params=aux_params)
        else:
            mod = mx.mod.Module(self.network, label_names=('label',), context=devs)
            mod.bind(data_shapes=data_shapes, label_shapes=label_shapes, for_training=True)
            mod.init_params(initializer=initializer)

        def reset_optimizer():
            mod.init_optimizer(kvstore='device',
                               optimizer="speechSGD",
                               optimizer_params={'lr_scheduler': lr,
                                                 'clip_gradient': clip_gradient,
                                                 'momentum': momentum,
                                                 'rescale_grad': 1.0},
                                                 # #0.015625 没有显示初始化，会导致rescale_grad被初始化为这个值，使得很难收敛；1/64
                                                 # 即1/batch_size
                                                 #'wd': weight_decay},
                                                 # 测试没有用wd的效果
                               force_init=True)

            # 使用这种方式初始化的optimiser，mse两三百！这种情况下需要设置rescale_grad为1/batch_size
            # optimizer = mx.optimizer.SGD(
            #     wd = 0.0005,
            #     momentum=0.9,
            #     clip_gradient = 5.0,
            #     lr_scheduler = lr)
            # mod.init_optimizer(optimizer=optimizer)

        reset_optimizer()
        warmup_epoch = 10
        last_acc = float("Inf")
        early_stop = False
        loss_increase_count = 0
        early_stop_num = 50
        for i_epoch in range(self.n_epoch):
            if loss_increase_count > early_stop_num:
                logging.info('early stop!!!')
                break
            tic = time.time()
            metric.reset()
            if i_epoch > warmup_epoch:
                lr.momentum = momentum
                # if lr.dynamic_lr > stop_factor_lr:
                #    lr.dynamic_lr = lr.dynamic_lr * 0.5
            for nbatch, data_batch in enumerate(train_dataiter):
                mod.forward(data_batch)
                mod.update_metric(metric, data_batch.label)
                # 根据准确率更改学习率，如果准确率没有提高则将学习率减半
                # 根据epoch更改momentum. 前十个阶段warming up阶段大学习率，小momentum。后面则开始momentum减半

                mod.backward()
                mod.update()
                batch_end_params = mx.model.BatchEndParam(epoch=i_epoch, nbatch=nbatch,
                                                          eval_metric=metric,
                                                          locals=None)
                for callback in batch_end_callbacks:
                    callback(batch_end_params)

            # name_value = metric.get_name_value() #似乎存在训练总mse远高于各个平均mse
            # for name, value in name_value:
            #     logging.info('Epoch[%d] train-%s=%f', i_epoch, name, value)
            toc = time.time()
            logging.info('Epoch[%d] Time cost=%.3f', i_epoch, toc - tic)
            train_dataiter.reset()

            # 在验证集合上判断优略。如果更好则保存
            metric.reset()
            val_dataiter.reset()
            for nbatch, data_batch in enumerate(val_dataiter):
                mod.forward(data_batch, is_train=False)
                mod.update_metric(metric, data_batch.label)

            curr_acc = None
            name_value = metric.get_name_value()
            for name, value in name_value:
                curr_acc = value
                logging.info('Epoch[%d] Validation-%s=%f', i_epoch, name, value)
            assert curr_acc is not None, 'cannot find Acc_exclude_padding in eval metric'
            if self.output_type == 'phoneme':
                curr_acc = 1 - curr_acc  # minize 1-acc

            if i_epoch > 0 and lr.dynamic_lr > stop_factor_lr and curr_acc > last_acc:
                loss_increase_count = loss_increase_count + 1
                logging.info('Epoch[%d] !!! Dev set performance drops, reverting this epoch',
                             i_epoch)
                logging.info('Epoch[%d] !!! LR decay: %g => %g', i_epoch,
                             lr.dynamic_lr, lr.dynamic_lr * lr_factor)

                lr.dynamic_lr *= lr_factor
                if lr.dynamic_lr < stop_factor_lr:
                    lr.dynamic_lr = stop_factor_lr
                # we reset the optimizer because the internal states (e.g. momentum)
                # might already be exploded, so we want to start from fresh
                reset_optimizer()
                mod.set_params(*last_params)
            if curr_acc < last_acc:
                loss_increase_count = 0
                last_params = mod.get_params()
                last_acc = curr_acc
                # save checkpoints
                mx.model.save_checkpoint(self.prefix, 0, mod.symbol, *last_params)
        if self.output_type == 'phoneme':
            logging.info('best acc: %f', 1 - last_acc)
        else:
            logging.info('best mse: %f', last_acc)

    def get_prefix(self):
        return self.prefix

def dnn_generation_mxnet(valid_file_list, prefix, n_ins, n_outs, out_file_list):
    # data_shapes = [('data', (self.batch_size, self.input_dim))]
    # sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, 0)
    # dnn_model = mx.mod.Module(sym, label_names=None, context=devs)
    # dnn_model.bind(data_shapes=data_shapes, for_training=False)
    # dnn_model.set_params(arg_params=arg_params, aux_params=aux_params)
    dnn_model = mx.model.FeedForward.load(prefix, 0)

    file_number = len(valid_file_list)
    for i in xrange(file_number):
        logging.info('generating %4d of %4d: %s' % (i + 1, file_number, valid_file_list[i]))
        fid_lab = open(valid_file_list[i], 'rb')
        features = np.fromfile(fid_lab, dtype=np.float32)
        fid_lab.close()
        features = features[:(n_ins * (features.size / n_ins))]
        test_set_x = features.reshape((-1, n_ins))
        predicted_parameter = dnn_model.predict(test_set_x)
        ### write to cmp file
        predicted_parameter = np.array(predicted_parameter, 'float32')
        print(os.path.basename(valid_file_list[i]), test_set_x.shape, predicted_parameter.shape)
        temp_parameter = predicted_parameter
        fid = open(out_file_list[i], 'wb')
        predicted_parameter.tofile(fid)
        logging.info('saved to %s' % out_file_list[i])
        fid.close()

