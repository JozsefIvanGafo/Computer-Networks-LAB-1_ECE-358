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
        arrival_packets = self.generate_exp_distribution(self.__lambda_param, 1000)
        arrival_packets.sort()

        # We generate the packet size for every packet that arrived
        length_packets = self.generate_exp_distribution(avg_len, 1000)

        # transmission time = packet's length / transmission rate of the link
        transmission_times = []
        for packet in length_packets:
            transmission_times.append(packet / trans_rate)

        #List where we will store all the departures times
        departure_list = []
        #Variable that tells us the time we are in the queue
        queue_time=0
        #Loop that calculates how the time of departure for each packet
        for count, arrival_packet_time in enumerate(arrival_packets):
    
            #If the queue is idle we just sum the arrival time +transmission [count] and we add it to the list
            if queue_time< arrival_packet_time:
                departure_time=arrival_packet_time+transmission_times[count]
                departure_list.append(departure_time)
            #Else there is a queue, and we add the last package  departure time (count-1)  + the transmission[count] 
            else:
                departure_time= departure_list[count-1]+transmission_times[count]
                departure_list.append(departure_time)
            #We update the queue time of the queue, with the las departured time
            queue_time=departure_time
        
        
        #We generate the observers
        observer_list = self.generate_exp_distribution(self.__lambda_param*5, 1000)
        observer_list.sort()

        #We join all the list while we add its type ("A"=arrival,"D"=departure and "O"=observer)
        """result_list = []
        for arrival_time in arrival_packets:
            result_list.append(["A", arrival_time])
        for departure_time in departure_list:
            result_list.append(["D", departure_time])
        for observer_time in observer_list:
            result_list.append(["O", observer_time])
        #We sort all the time events by time
        sorted_list = sorted(result_list, key=lambda x: x[1])

        num_idle = 0
        avg_packets_in_queue = []"""
        """
        # FIXME:
        for count, element,  in enumerate(sorted_list):
            if element[0]=="A":
                num_arrival+=1
            elif element[0]=="D":
                num_departed+=1
            else:
                if num_arrival-num_departed== 0:
                    num_idle += 1
                else:
                    avg_packets_in_queue.append(num_arrival-num_departed)
                    
                num_observers+=1
                #TODO: E[N] = time average of packets in the queue
                #TODO:PIDLE = The proportion of time the server is idle, i.e., no packets in the queue nor a packet is being transmitted
                #TODO:PLOSS = As it is infinite we don't lose packets
                """
        
        #total_avg = 0 # E[N]
        #for e in avg_packets_in_queue:
        #    total_avg += e
        #total_avg = total_avg/len(avg_packets_in_queue)


        # while sorted_list[i][1]< 1:
        #     if sorted_list[i][0] == "O" :
        #         break
        avg_packets_in_queue=[]
        sorted_list=[]
        num_idle=0
        num_arrival = 0
        num_departed = 0
        num_observers = 0
        while num_arrival<1000 and num_departed<1000 and num_observers<1000:
            #We register the time of every list
            arrival=arrival_packets[num_arrival]
            observers=observer_list[num_observers]
            departed=departure_list[num_departed]
            #Conditionals
            if arrival<min(observers,departed):
                num_arrival+=1
                sorted_list.append(["A",arrival])
            elif departed <min(arrival,observers):
                num_departed+=1
                sorted_list.append(["D",departed])
            else:
                if num_arrival-num_departed== 0:
                    num_idle += 1
                else:
                    avg_packets_in_queue.append(num_arrival-num_departed)
                num_observers+=1
                sorted_list.append(["O",observers])
                
        total_avg = 0 # E[N]
        for e in avg_packets_in_queue:
            total_avg += e
        total_avg = total_avg/len(avg_packets_in_queue)
        
        

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
trans_rate=1000*1000000
a.m_m_1_queue(2000,trans_rate)