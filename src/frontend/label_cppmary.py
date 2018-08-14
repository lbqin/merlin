#!/usr/bin/python
#coding=utf-8

import logging
import imp
import numpy
import sys
sys.path.append('..')
from io_funcs.binary_io import BinaryIOCollection

from lxml import etree

from frontend.label_normalisation import HTSLabelNormalisation

# context-dependent printing format for Numpy - should move this out to a utility file somewhere
import contextlib
import os
import re

@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = numpy.get_printoptions()
    numpy.set_printoptions(*args, **kwargs)
    yield
    numpy.set_printoptions(**original)


class LabelCppmary(object):

    def __init__(self,path=""):
        self.logger = logging.getLogger("labels")
        self.configuration = None
        self.label_dimension = None
        self.confpath = path

    def prepare_label_feature(self, ori_file_list, output_file_list,ivector=0):
        print("prepare cppmary in_duration feature")
        logger = logging.getLogger("dur")
        utt_number = len(ori_file_list)
        if utt_number != len(output_file_list):
            print("the number of input and output files should be the same!\n")
            sys.exit(1)
        for i in xrange(utt_number):
            if ivector == 1:
                self.extract_label_features_extend(ori_file_list[i], output_file_list[i])
            else:
                self.extract_label_features(ori_file_list[i], output_file_list[i])

    def prepare_acoustic_label_feature(self, ori_file_list, output_file_list,ivector=0):
        print("prepare cppmary in_acoustic feature")
        logger = logging.getLogger("acoustic")
        utt_number = len(ori_file_list)
        if utt_number != len(output_file_list):
            print("the number of input and output files should be the same!\n")
            sys.exit(1)
        for i in xrange(utt_number):
            if ivector ==1:
                self.extract_acousitc_label_features_extend(ori_file_list[i], output_file_list[i])
            else:
                self.extract_acousitc_label_features(ori_file_list[i], output_file_list[i])

    def prepare_dur_feature(self, ori_file_list, output_file_list):
        print("prepare cppmary out_duration feature")
        logger = logging.getLogger("dur")
        utt_number = len(ori_file_list)
        if utt_number != len(output_file_list):
            print("the number of input and output files should be the same!\n")
            sys.exit(1)
        for i in xrange(utt_number):
            self.extract_dur_features(ori_file_list[i], output_file_list[i])

    def prepare_predict_label(self, orig_label_file_list, gen_label_file_list, predict_duration_file_list):
        print("prepare cppmary prepare_predict_label feature")
        logger = logging.getLogger("dur")
        utt_number = len(orig_label_file_list)
        if utt_number != len(predict_duration_file_list):
            print("the number of input and output files should be the same!\n")
            sys.exit(1)
        for i in xrange(utt_number):
            self.compose_predict_label(orig_label_file_list[i], gen_label_file_list[i], predict_duration_file_list[i])

    def prepare_predict_label_pure(self, orig_pure_label_file_list, lab_dim, gen_label_file_list, predict_duration_file_list):
        logger = logging.getLogger("dur")
        utt_number = len(orig_pure_label_file_list)
        if utt_number != len(predict_duration_file_list):
            print("the number of input and output files should be the same!\n", utt_number, len(predict_duration_file_list))
            sys.exit(1)
        for i in xrange(utt_number):
            self.compose_predict_label_pure(orig_pure_label_file_list[i], lab_dim , gen_label_file_list[i], predict_duration_file_list[i])

    def compose_predict_label(self, orig_label_file, gen_label_file, predict_duration_file):
        io_funcs = BinaryIOCollection()
        origMat = io_funcs.file2matrix(orig_label_file)

        state_number = 5
        duration, in_frame_number = io_funcs.load_binary_file_frame(predict_duration_file, state_number)
        print(orig_label_file, origMat.shape, duration.shape)
        assert origMat.shape[0] == in_frame_number
        origMat[:, -5:] = duration
        origMat = origMat.astype(int)
        io_funcs.matrix2file(origMat, gen_label_file)

    def compose_predict_label_pure(self, orig_label_file, lab_dim, gen_label_file, predict_duration_file):
        io_funcs = BinaryIOCollection()
        origMat, lab_frame_number = io_funcs.load_binary_file_frame(orig_label_file, lab_dim)
        state_number = 5
        duration, in_frame_number = io_funcs.load_binary_file_frame(predict_duration_file, state_number)
        print(orig_label_file, predict_duration_file, origMat.shape, duration.shape)
        assert(lab_frame_number == in_frame_number)
        origMat = numpy.concatenate((origMat, duration), axis=1)
        print(origMat.shape)
        origMat = origMat.astype(int)
        io_funcs.matrix2file(origMat, gen_label_file)

    def extract_dur_features(self, orig_file, output_file):
        io_funcs = BinaryIOCollection()
        totalMat = io_funcs.file2matrix(orig_file)
        self.label_dimension = totalMat.shape[1] - 5  # collum num
        durMat = totalMat[:, -5:]

        io_funcs.array_to_binary_file(durMat, output_file)

    def check_path_style(self,orig_file):
        ori_basestr = os.path.basename(orig_file)
        ori_baseSplit = ori_basestr.split('_')
        spker="tai"
        gender=1
        emot=1

        if len(ori_baseSplit) == 5:
            spker = ori_baseSplit[3]
            if ori_baseSplit[3] == 'hfn':
                emot = 2
                gender = 2
            else:
                emot = 1
                if ori_baseSplit[3] in ('qiezi','tai','mmt','bai','hmbb','yingguo','yuxiang'):
                    gender = 1
                else:
                    gender = 2
        elif len(ori_baseSplit) == 2 or len(ori_baseSplit) == 3:
            emot = 1
            spker = ori_baseSplit[0]
            if ori_baseSplit[0] in ('qiezi', 'tai', 'mmt', 'bai', 'hmbb', 'yingguo', 'yuxiang'):
                gender = 1
            else:
                gender = 2
        else:
            emot = 1

        return  spker,emot,gender

    def extract_label_features(self, orig_file, output_file):
        io_funcs = BinaryIOCollection()
        totalMat = io_funcs.file2matrix(orig_file)
        self.label_dimension = totalMat.shape[1] - 5  # collum num
        labelMat = totalMat[:, :-5]
        #print orig_file, totalMat.shape, labelMat.shape

        io_funcs.array_to_binary_file(labelMat, output_file)

    # 将最后5维最为时长进行扩展，并增加特征。lab维度变化：-5 + 9
    def extract_acousitc_label_features(self, orig_file, output_file):
        io_funcs = BinaryIOCollection()
        totalMat = io_funcs.file2matrix(orig_file, numpy.int)
        labelMat = totalMat[:, :-5]
        durMat = totalMat[:, -5:]

        label_len = totalMat.shape[1] - 5

        self.label_dimension = label_len + 9

        phone_number = labelMat.shape[0]

        label_feature_matrix = numpy.empty((100000, self.label_dimension))

        state_number = 5
        label_feature_index = 0

        for phone_index in xrange(phone_number) :
            label_vector = labelMat[phone_index,:]
            state_vector = durMat[phone_index, :]
            phone_duration = 0
            state_duration_bases = numpy.zeros((5,), dtype=numpy.int)
            for state_index in xrange(state_number):
                state_duration_bases[state_index] = phone_duration
                phone_duration = phone_duration + state_vector[state_index]

            for state_index in xrange(state_number):
                frame_number = state_vector[state_index]
                current_block_binary_array = numpy.zeros((frame_number, self.label_dimension))
                state_duration_base = state_duration_bases[state_index]
                state_index_backward = state_number - state_index
                for i in xrange(frame_number):
                    current_block_binary_array[i, 0:label_len] = label_vector

                    current_block_binary_array[i, label_len] = float(i + 1) / float(
                        frame_number)  ## fraction through state (forwards)
                    current_block_binary_array[i, label_len + 1] = float(frame_number - i) / float(
                        frame_number)  ## fraction through state (backwards)
                    current_block_binary_array[i, label_len + 2] = float(frame_number)  ## length of state in frames
                    current_block_binary_array[i, label_len + 3] = float(state_index)  ## state index (counting forwards)
                    current_block_binary_array[i, label_len + 4] = float(
                        state_index_backward)  ## state index (counting backwards)

                    current_block_binary_array[i, label_len + 5] = float(phone_duration)  ## length of phone in frames
                    current_block_binary_array[i, label_len + 6] = float(frame_number) / float(
                        phone_duration)  ## fraction of the phone made up by current state
                    current_block_binary_array[i, label_len + 7] = float(phone_duration - i - state_duration_base) / float(
                        phone_duration)  ## fraction through phone (forwards)
                    current_block_binary_array[i, label_len + 8] = float(state_duration_base + i + 1) / float(
                        phone_duration)  ## fraction through phone (backwards)

                label_feature_matrix[label_feature_index:label_feature_index + frame_number, ] = current_block_binary_array
                label_feature_index = label_feature_index + frame_number

        label_feature_matrix = label_feature_matrix[0:label_feature_index, ]

        #print label_feature_matrix.shape

        io_funcs.array_to_binary_file(label_feature_matrix, output_file)
        #fid = open(output_file, 'w')
        #label_feature_matrix.tofile(fid, sep=' ')
        #fid.close()


    def open_spk_ivector_features(self,spk_file):
        spk_file2 = self.confpath + '/spk_ivector.ark' #'/home/zhiming/speech/deeptts/egs/world/s1/conf/spk_ivector.ark'
        assert os.path.isfile(spk_file2) == True
        #print spk_file2
        f2 = open(spk_file2, 'r')
        spk_features = f2.readlines()
        f2.close()

        features_lines = {}
        nums = 0
        for line in spk_features:
            spkft = line.decode('utf-8').strip()
            ift = re.findall('\[ (.*?) \]', spkft)
            #print(ift)
            #print spkft
            tmp_ft = spkft.split(" ")
            if len(tmp_ft) == 0:
                print("empty line")
            else:
                nums = nums + 1
                features_lines[tmp_ft[0]] = ift[0]
        return features_lines

    def extract_label_features_extend(self, orig_file, output_file):
        spk,emot,gender = self.check_path_style(orig_file)
        features_spk = self.open_spk_ivector_features("")
        featuresList=[]
        if spk == 'hfn':
            featuresList = features_spk['hfnn'].split(" ")
        else:
            featuresList = features_spk[spk].split(" ")
        assert len(featuresList) == 30
        #print spk,orig_file
        io_funcs = BinaryIOCollection()
        totalMat = io_funcs.file2matrix(orig_file)
        self.label_dimension = totalMat.shape[1] - 5 + 2 + 30 # collum num
        labelMat = totalMat[:, :-3]
        labelMat[:, -2] = emot
        labelMat[:, -1] = gender
        labelMatNew = numpy.c_[labelMat,labelMat.shape[0]*[featuresList]]
        #print orig_file, output_file,totalMat.shape, labelMatNew.shape

        io_funcs.array_to_binary_file(labelMatNew, output_file)

    # 将最后5维最为时长进行扩展，并增加特征。lab维度变化：-5 + 9
    def extract_acousitc_label_features_extend(self, orig_file, output_file):
        spk, emot,gender = self.check_path_style(orig_file)
        features_spk = self.open_spk_ivector_features("")
        featuresList = []
        if spk == 'hfn':
            featuresList = features_spk['hfnn'].split(" ")
        else:
            featuresList = features_spk[spk].split(" ")
        assert len(featuresList) == 30
        #print spk, orig_file
        io_funcs = BinaryIOCollection()
        totalMat = io_funcs.file2matrix(orig_file, numpy.int)
        labelMat = totalMat[:, :-5]
        durMat = totalMat[:, -5:]

        label_len = totalMat.shape[1] - 5

        self.label_dimension = label_len + 9 + 2 + 30

        phone_number = labelMat.shape[0]

        label_feature_matrix = numpy.empty((100000, self.label_dimension))

        state_number = 5
        label_feature_index = 0

        for phone_index in xrange(phone_number) :
            label_vector = labelMat[phone_index,:]
            state_vector = durMat[phone_index, :]
            phone_duration = 0
            state_duration_bases = numpy.zeros((5,), dtype=numpy.int)
            for state_index in xrange(state_number):
                state_duration_bases[state_index] = phone_duration
                phone_duration = phone_duration + state_vector[state_index]

            for state_index in xrange(state_number):
                frame_number = state_vector[state_index]
                current_block_binary_array = numpy.zeros((frame_number, self.label_dimension))
                state_duration_base = state_duration_bases[state_index]
                state_index_backward = state_number - state_index
                for i in xrange(frame_number):
                    current_block_binary_array[i, 0:label_len] = label_vector

                    #current_block_binary_array[i, label_len] = spk
                    #current_block_binary_array[i, label_len+1] = emot

                    current_block_binary_array[i, label_len ] = emot
                    current_block_binary_array[i, label_len+1] = gender
                    current_block_binary_array[i, label_len+2:label_len+32] = featuresList

                    current_block_binary_array[i, label_len+32] = float(i + 1) / float(
                        frame_number)  ## fraction through state (forwards)
                    current_block_binary_array[i, label_len +32 + 1] = float(frame_number - i) / float(
                        frame_number)  ## fraction through state (backwards)
                    current_block_binary_array[i, label_len +32 + 2] = float(frame_number)  ## length of state in frames
                    current_block_binary_array[i, label_len +32 + 3] = float(state_index)  ## state index (counting forwards)
                    current_block_binary_array[i, label_len +32 + 4] = float(
                        state_index_backward)  ## state index (counting backwards)

                    current_block_binary_array[i, label_len +32 + 5] = float(phone_duration)  ## length of phone in frames
                    current_block_binary_array[i, label_len +32 + 6] = float(frame_number) / float(
                        phone_duration)  ## fraction of the phone made up by current state
                    current_block_binary_array[i, label_len +32 + 7] = float(phone_duration - i - state_duration_base) / float(
                        phone_duration)  ## fraction through phone (forwards)
                    current_block_binary_array[i, label_len +32 + 8] = float(state_duration_base + i + 1) / float(
                        phone_duration)  ## fraction through phone (backwards)

                label_feature_matrix[label_feature_index:label_feature_index + frame_number, ] = current_block_binary_array
                label_feature_index = label_feature_index + frame_number

        label_feature_matrix = label_feature_matrix[0:label_feature_index, ]

        #print label_feature_matrix.shape

        io_funcs.array_to_binary_file(label_feature_matrix, output_file)
        #fid = open(output_file, 'w')
        #label_feature_matrix.tofile(fid, sep=' ')
        #fid.close()


if __name__ == '__main__':
    logger = logging.getLogger("labels")
    logger.setLevel(logging.DEBUG)
    # a console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    label_cppmary = LabelCppmary()
    label_cppmary.extract_acousitc_label_features("../test.lab", "../test_compose.lab")

