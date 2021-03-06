#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class Synapse(object):

    def __init__(self, start, end, weight):
        self.start = start  # Neuron
        self.end = end  # Neuron
        self.weight = weight  # Float

    def to_string(self, neuron):
        direction = "OUT" if self.start == neuron else "IN"
        string = ""
        string += "\n         + " + direction + " | W"
        string += self.end.id + self.start.id + ": " + str(self.weight)

        return string
