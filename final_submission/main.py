"""
University of Waterloo Fall 2023 ECE-358 LAB-1  Group 151
József IVÁN GAFO (21111635) jivangaf@uwaterloo.ca
Sonia NAVAS RUTETE (21111397) srutete@uwaterloo.ca
V 1:0
In this module we will write the main code
"""

# Imports
from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import os


# We define the type of events
ARRIVAL = "A"
DEPARTURE = "D"
OBSERVER = "O"


class Lab1():
    """
    This class is in charge of containing all the 
    methods required for the lab 1
    """

    def __init__(self) -> None:
        pass

    # We write the main code
    def question1(self, lambda_param):
        """
        This method is in charge of generating 1000 random variables 
        with lambda 75 with numpy library and it prints the mean and variance
        @param lambda_param: Value for lambda 
        @return : None
        """
        # For this exercise we will use the library numpy
        # as it makes the operations easier

        # Calculate the expected mean and variance for an exponential distribution with λ=75
        expected_mean = 1 / lambda_param
        expected_variance = 1 / (lambda_param ** 2)

        # We generate 1000 exponential random variables
        generated_numbers = [self.__generate_exp_distribution(
            lambda_param) for _ in range(1000)]

        # We do the mean and the variance
        mean = np.mean(generated_numbers)
        variance = np.var(generated_numbers)

        # Print the results
        print("\nQUESTION 1:")
        print(f"\tMean of generated random variables: {mean}")
        print(f"\tExpected mean for λ=75: {expected_mean}")
        print(f"\tVariance of generated random variables: {variance}")
        print(f"\tExpected variance for λ=75: {expected_variance}\n")

    def m_m_1_queue(self, avg_len: int, trans_rate: int, lambda_par: int, T: int) -> list:
        """
        Build your simulator for this queue and explain in words what you have done. Show your code in the report. In 
        particular, define your variables. Should there be a need, draw diagrams to show your program structure. Explain how you 
        compute the performance metrics. Type of queue M/M/ (infinite queue)
        @param avg_len: Is the average length of a packet in bits (L)
        @param trans_rate: Is the transmission rate of the output link in bits/sec (C)
        @return: None
        """
        # ! Declaration of variables for the m_m_1 queue
        num_arrival = 0
        num_departed = 0
        num_observers = 0
        transmission_times = []
        arrival_list = []
        departure_list = []
        observer_list = []
        event_list = []
        result_list = []

        # * Arrival

        # we generate the arrivals
        arrival_list = self.__generate_mm1_arr_obs(lambda_par, T)
        

        # * observers
        # Now we add the observers event
        observer_list = self.__generate_mm1_arr_obs(lambda_par, T, 5)


        
        

        # * Departure
        # * generate packet lengths for each arrival
        length_packets = []
        # We create the packet size for each arrival
        length_arrival = len(arrival_list)
        length_packets = [self.__generate_exp_distribution(1/avg_len) for _ in range(length_arrival)]

        # * Calculate how much time takes to process all the packet

        for packet in length_packets:
            transmission_times.append(packet / trans_rate)

        # *We calculate the departure time
        queue_time = 0
        departure_time = 0
        # Loop that calculates how the time of departure for each packet
        for count, arrival_packet_time in enumerate(arrival_list):
            # If the queue is idle we just sum the arrival time +transmission [count] and we add it to the list
            if queue_time < arrival_packet_time:
                departure_time = arrival_packet_time+transmission_times[count]    
            # Else there is a queue, and we add the last package  departure time (count-1)  + the transmission[count]
            else:
                departure_time = departure_list[count-1] + transmission_times[count]
            departure_list.append(departure_time)
            # We update the queue time of the queue, with the las departure time
            queue_time = departure_time

        
        # * Order all packets by time with its type
        for arrival_time in arrival_list:
            result_list.append(["A", arrival_time])
        for departure_time in departure_list:
            result_list.append(["D", departure_time])
        for observer_time in observer_list:
            result_list.append(["O", observer_time])
        # We sort all the time events by time
        event_list = sorted(result_list, key=lambda x: x[1])



        # * We calculate E[n] and p_idle
        # Declaration of variables
        total_num_packs_queue = 0
        total_observer_idles = 0

        for _, event in enumerate(event_list):
            # for i in range(len(event_list)):
            #event_type= event_list.pop(0)
            event_type = event[0]
            # Arrival
            if event_type == 'A':
                num_arrival += 1
            elif event_type == 'D':
                num_departed += 1
            else:
                num_observers += 1
                # We record the num of packets that are currently on the queue
                total_num_packs_queue += (num_arrival-num_departed)
                if num_arrival == num_departed:
                    total_observer_idles += 1

        return total_num_packs_queue/num_observers, total_observer_idles/num_observers
    
    def __generate_mm1_arr_obs(self, lambda_par, T, steps=1):
        """
        This method is in charge of generating the list of arrivals and observers
        @param lambda_param: An integer that contains the average number of packets arrived
        @param T: Duration of the simulation
        @param steps: It is 1 for default, will change for observers as the param is different
        @return list: We return a list with the events generated
        """
        # Iteration variables
        aux_list = []
        simulation_time = 0
        #Loop we iterate until we reach the simulation time
        while simulation_time < T:
            #We generate an exponential distribution
            arrival_timestamp = self.__generate_exp_distribution(lambda_par*steps)+simulation_time
            #We add to the aux list
            aux_list.append(arrival_timestamp)
            #We update the simulation time with the new arrival
            simulation_time = arrival_timestamp
        #We return the aux list
        return aux_list

    def __generate_exp_distribution(self, lambda_param: int) -> list:
        """
        This method is in charge of generating exponential random variables
        @param lambda_param: An integer that contains the average number of packets arrived
        @param size: An integer that defines how many numbers we generate
        @return list: We return a list with the numbers generated
        """
        expected_mean = 1/lambda_param
        return -expected_mean*math.log(1-random.random())

    # #     self.__event_list.append((type, timestamp))
    def generate_point(self, i, avg_len, trans_rate, lambda_par, T):
        # Calculate data point for a specific 'i'
        list_m_m_1 = self.m_m_1_queue(avg_len, trans_rate, lambda_par, T)
        # If we want E[n] then type_info is 0 if is p_idle then type_info is 1
        return [i, list_m_m_1]

    def create_graph_for_m_m_1_queue(self, avg_len, trans_rate, T):
        step = 0.1
        start = 0.25
        end = 0.95
        # Graph for E[N]
        result=[]
        print("generating points for graph mm1 queue")
        #cores=4
        with Pool() as pool:
            input_data = [(i, avg_len, trans_rate, trans_rate * i / avg_len, T)
                        for i in np.arange(start, end, step)]
            pool_list = pool.starmap(self.generate_point, input_data)
            print(pool_list)
        print("Finished generating points for graph mm1 queue ")
        result.append(pool_list)
        print(len(result))


        # We save the points
        #pos 0 is for E[n] and pos 1 is for p_idle
        x = [point[0] for point in result[0]]
        y = [[point[1][0] for point in result[0]],
        [point[1][1] for point in result[0]]]

        #We create the graph
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        graph_list=[ax1,ax2]
        text=[['Average number in system E[n]',"average number of packets in the queue as a function of p"],
              ["Average number in system p_idle","The proportion of time the system is idle as a function of p"]]
        for i,graph in enumerate(graph_list):
            # We initialize the graph
            graph.plot(x, y[i],label="")
            graph.set_xlabel('Traffic intensity p')
            graph.set_ylabel(text[i][0])
            graph.set_title(text[i][1])
        #We save it
        # Save the figure as an image in the "graphs" folder
        script_directory = os.path.dirname(__file__)

        # Save the figure as an image named "exercise_3.png" in the same folder as the script
        image_path = os.path.join(script_directory, 'exercise_3.png')
        plt.savefig(image_path)

def check_T(avg_len, trans_rate, lambda_par, T):

    a = Lab1()
    T_counter = 1
    percentage = 0.05
    dif_count_E = 100
    dif_count_pidle = 100
    final_T = 1

    gate = True


    while gate:
        E, pidle = a.m_m_1_queue(avg_len, trans_rate, lambda_par, T_counter*T)
        E2, pidle2 = a.m_m_1_queue(avg_len, trans_rate, lambda_par, (T_counter+1)*T)
        # print("E is the folowing: " + str(E))
        # print("P is the folowing: " + str(pidle))
        # print("\n")
        # print("E2 is the folowing: " + str(E2))
        # print("P2 is the folowing: " + str(pidle2))

        difference_E = abs(E-E2)
        difference_pidle = abs(pidle-pidle2)
        if(difference_E <= E*percentage and difference_pidle <= pidle*percentage):
            print(T_counter+1)
            if dif_count_E > difference_E and dif_count_pidle > difference_pidle:
                final_T = T_counter+1
                dif_count_E = difference_E
                dif_count_pidle = difference_pidle
            else:
                gate = False
        T_counter += 1
        # print("Final T: " + str(final_T))
    
    return final_T
    
if __name__ == "__main__":
    a = Lab1()
    lambda_par = 75
    trans_rate = 1_000_000
    avg_packet_length = 2_000
    T = 1_000

    # RUNNING THE LAB
    #QUESTION 1
    #a.question1(lambda_par)
#
    ## INFINITE QUEUE
    #print("QUESTION 2:\n")
    #X = check_T(avg_packet_length, trans_rate, lambda_par, T)
    #print("\tThe Final T will be: " + str(T*X))

    print("QUESTION3:\n")
    print("the graph will be generated in exercise3.png")
    a.create_graph_for_m_m_1_queue(avg_packet_length,trans_rate,2*T)

    # For p=1.2
    E, pidle = a.m_m_1_queue(avg_packet_length, trans_rate, trans_rate * 1.2 / avg_packet_length, 2*T)
    print("QUESTION 4:\n")
    print("\tFor p = 1.2, the value of E[N] = "+ str(E) + " and the value of pidle =" + str(pidle))


    # FINITE
    #a.m_m_1_k_queue(avg_packet_length,trans_rate,lambda_par,T,10)
    """k = [10, 25, 50]
    for element in k:
        print(generate_graph_points2(avg_packet_length, trans_rate, T, element))"""
    #a.create_graph_for_m_m_1_k_queue(avg_packet_length,trans_rate,T)
    
   #""" print(a.m_m_1_k_queue(avg_packet_length, trans_rate, lambda_par, T, 10))
    #print(a.m_m_1_k_queue(avg_packet_length, trans_rate, lambda_par, 2*T, 10))
    #print(a.m_m_1_k_queue(avg_packet_length, trans_rate, lambda_par, 3*T, 10))"""
