"""
University of Waterloo Fall 2023 ECE-358 LAB-1  Group 151
József IVÁN GAFO (21111635) jivangaf@uwaterloo.ca
Sonia NAVAS RUTETE (21111397) srutete@uwaterloo.ca
V 1:0
In this module we will write the main code
"""

# Imports
import math
import random
import numpy as np


class Lab1():
    """
    This class is in charge of containing all the 
    methods required for the lab 1
    """

    def __init__(self, lambda_param) -> None:
        self.__lambda_param = lambda_param

    # We write the main code
    def question1(self):
        """
        This method is in charge of generating 1000 random variables 
        with lambda 75 with numpy library and it prints the mean and variance
        @param : None
        @return : None
        """
        # For this exercise we will use the library numpy
        # because it makes the operations easier

        # Calculate the expected mean and variance for an exponential distribution with λ=75
        expected_mean = 1 / self.__lambda_param
        expected_variance = 1 / (self.__lambda_param ** 2)

        # We generate 1000 exponential random variables
        generated_numbers = self.generate_exp_distribution(
            self.__lambda_param, 1000)

        mean = np.mean(generated_numbers)
        variance = np.var(generated_numbers)

        # Print the results
        print(f"Mean of generated random variables: {mean}")
        print(f"Expected mean for λ=75: {expected_mean}")
        print(f"Variance of generated random variables: {variance}")
        print(f"Expected variance for λ=75: {expected_variance}")

    def m_m_1_queue(self, avg_len: float, trans_rate: float) -> None:
        """
        Build your simulator for this queue and explain in words what you have done. Show your code in the report. In 
        particular, define your variables. Should there be a need, draw diagrams to show your program structure. Explain how you 
        compute the performance metrics. Type of queue M/M/ (infinite queue)
        @param avg_len: Is the average length of a packet in bits (L)
        @param trans_rate: Is the transmission rate of the output link in bits/sec (C)
        @return: None
        """

        # TODO: QUESTION Q1 okay?
        # TODO: QUESTION simulation time T

        # Define variables for the queue
        num_arrival = 0
        num_departed = 0
        num_observers = 0

        # generate random variables for packet arrival

        # We generate 1000 exponential random variables
        # that represent the times that the packets arrived
        arrival_packets = self.generate_exp_distribution(
            self.__lambda_param, 1000)
        arrival_packets.sort()

        # We generate the packet size for every packet that arrived
        length_packets = self.generate_exp_distribution(avg_len, 1000)

        # transmission time = packet's length / transmission rate of the link
        transmission_times = []
        for packet in length_packets:
            transmission_times.append(packet / trans_rate)

        # TODO:packet departure
        # FIXME:
        departure_list = []
        for count, arrival_packets in enumerate(arrival_packets):
            if num_arrival == num_departed:
                queue_idle = True
            if queue_idle:
                # if queue is idle: departure of packets= arrival + transmission of i
                # departure_list.append(arrival_packets[element]+transmission_times[element])
                departure_list.append(
                    arrival_packets+transmission_times[count])
            else:
                # if queue has packets: departure of packet= departure of i-1 + transmission of i
                departure_list.append(
                    departure_list[count-1]+transmission_times[count])

        # TODO: Observer(monitor the queue)
        # number of observers =
        observer_list = self.generate_exp_distribution(
            self.__lambda_param*5, 5)

        # TODO:Sort all the events

    # Auxiliary methods
    def generate_exp_distribution(self, lambda_param: int, size: int) -> list:
        """
        This method is in charge of generating exponential random variables
        @param lambda_param: An integer that contains the average number of packets arrived
        @param size: An integer that defines how many numbers we generate
        @return list: We return a list with the numbers generated
        """
        # Define variables to generate the numbers
        expected_mean = 1/lambda_param
        generated_numbers = []
        # Generate  size exponential random numbers
        for _ in range(size):
            # Generate a number between [0,1)
            random_number = random.random()
            # We add the new generated number to the list
            generated_numbers.append(-expected_mean*math.log(1-random_number))
        return generated_numbers


a = Lab1(75)
a.question1()


