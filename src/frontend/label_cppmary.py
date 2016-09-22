#!/usr/bin/python
#coding=utf-8

import logging
import imp
import numpy
import sys
sys.path.append('..')
from io_funcs.binary_io import BinaryIOCollection

from lxml import etree

from frontend.label_normalisation import HTSLabelNormalisation, XMLLabelNormalisation

# context-dependent printing format for Numpy - should move this out to a utility file somewhere
import contextlib


@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = numpy.get_printoptions()
    numpy.set_printoptions(*args, **kwargs)
    yield
    numpy.set_printoptions(**original)


class LabelCppmary(object):

    def __init__(self):
        self.logger = logging.getLogger("labels")
        self.configuration = None
        self.label_dimension = None

    def prepare_label_feature(self, ori_file_list, output_file_list):
        print "prepare cppmary in_duration feature"
        logger = logging.getLogger("dur")
        utt_number = len(ori_file_list)
        if utt_number != len(output_file_list):
            print "the number of input and output files should be the same!\n";
            sys.exit(1)
        for i in xrange(utt_number):
            self.extract_label_features(ori_file_list[i], output_file_list[i])

    def prepare_acoustic_label_feature(self, ori_file_list, output_file_list):
        print "prepare cppmary in_acoustic feature"
        logger = logging.getLogger("acoustic")
        utt_number = len(ori_file_list)
        if utt_number != len(output_file_list):
            print "the number of input and output files should be the same!\n";
            sys.exit(1)
        for i in xrange(utt_number):
            self.extract_acousitc_label_features(ori_file_list[i], output_file_list[i])

    def prepare_dur_feature(self, ori_file_list, output_file_list):
        print "prepare cppmary out_duration feature"
        logger = logging.getLogger("dur")
        utt_number = len(ori_file_list)
        if utt_number != len(output_file_list):
            print "the number of input and output files should be the same!\n";
            sys.exit(1)
        for i in xrange(utt_number):
            self.extract_dur_features(ori_file_list[i], output_file_list[i])

    def prepare_predict_label(self, orig_label_file_list, gen_label_file_list, predict_duration_file_list):
        print "prepare cppmary prepare_predict_label feature"
        logger = logging.getLogger("dur")
        utt_number = len(orig_label_file_list)
        if utt_number != len(predict_duration_file_list):
            print "the number of input and output files should be the same!\n";
            sys.exit(1)
        for i in xrange(utt_number):
            self.compose_predict_label(orig_label_file_list[i], gen_label_file_list[i], predict_duration_file_list[i])

    def compose_predict_label(self, orig_label_file, gen_label_file, predict_duration_file):
        io_funcs = BinaryIOCollection()
        origMat = io_funcs.file2matrix(orig_label_file)

        state_number = 5
        duration, in_frame_number = io_funcs.load_binary_file_frame(predict_duration_file, state_number)
        assert origMat.shape[0] == in_frame_number
        origMat[:, -5:] = duration
        origMat = origMat.astype(int)
        io_funcs.matrix2file(origMat, gen_label_file)

    def extract_dur_features(self, orig_file, output_file):
        io_funcs = BinaryIOCollection()
        totalMat = io_funcs.file2matrix(orig_file)
        self.label_dimension = totalMat.shape[1] - 5  # collum num
        durMat = totalMat[:, -5:]

        io_funcs.array_to_binary_file(durMat, output_file)

    def extract_label_features(self, orig_file, output_file):
        io_funcs = BinaryIOCollection()
        totalMat = io_funcs.file2matrix(orig_file)
        self.label_dimension = totalMat.shape[1] - 5  # collum num
        labelMat = totalMat[:, :-5]
        print orig_file, totalMat.shape, labelMat.shape

        io_funcs.array_to_binary_file(labelMat, output_file)

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

        print label_feature_matrix.shape

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

