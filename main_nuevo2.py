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
        @param : None
        @return : None
        """
        # For this exercise we will use the library numpy
        # because it makes the operations easier

        # Calculate the expected mean and variance for an exponential distribution with λ=75
        expected_mean = 1 / lambda_param
        expected_variance = 1 / (lambda_param ** 2)

        # We generate 1000 exponential random variables
        generated_numbers = [self.__generate_exp_distribution(
            lambda_param) for _ in range(1000)]
        mean = np.mean(generated_numbers)
        variance = np.var(generated_numbers)

        # Print the results
        print(f"Mean of generated random variables: {mean}")
        print(f"Expected mean for λ=75: {expected_mean}")
        print(f"Variance of generated random variables: {variance}")
        print(f"Expected variance for λ=75: {expected_variance}")

    def m_m_1_queue(self, avg_len: int, trans_rate: int, lambda_par: int, T: int) -> None:
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
        # Variable for iterations
        simulation_time = 0
        # we generate the arrivals
        while simulation_time < T:
            arrival_timestamp = self.__generate_exp_distribution(
                lambda_par)+simulation_time
            arrival_list.append(arrival_timestamp)
            simulation_time = arrival_timestamp
        arrival_list.sort()
        length_packets = []

        # We create the packet size for each arrival
        length_arrival=len(arrival_list)
        length_packets=[self.__generate_exp_distribution(1/avg_len) for _ in range(length_arrival)]
        #for _ in range(len(arrival_list)):
            # FIXME: COMPROBAR QUE ESTE BIEN

           # length_packets.append(self.__generate_exp_distribution(1/avg_len))

        # How much time it takes to process each packet
        
        for packet in length_packets:
            transmission_times.append(packet / trans_rate)

        # * Departure
        # We add the departure time
        queue_time = 0
        departure_time = 0
        # Loop that calculates how the time of departure for each packet
        for count, arrival_packet_time in enumerate(arrival_list):
            # If the queue is idle we just sum the arrival time +transmission [count] and we add it to the list
            if queue_time < arrival_packet_time:
                departure_time = arrival_packet_time+transmission_times[count]
                departure_list.append(departure_time)
            # Else there is a queue, and we add the last package  departure time (count-1)  + the transmission[count]
            else:
                departure_time = departure_list[count-1] + \
                    transmission_times[count]
                departure_list.append(departure_time)
            # We update the queue time of the queue, with the las departured time
            queue_time = departure_time

        # Now we add the observers event
        simulation_time = 0
        while simulation_time < T:
            observation_timestamp = self.__generate_exp_distribution(lambda_par*5)+simulation_time
            observer_list.append(observation_timestamp)
            simulation_time = observation_timestamp

            

        
        for arrival_time in arrival_list:
            result_list.append(["A", arrival_time])
        for departure_time in departure_list:
            result_list.append(["D", departure_time])
        for observer_time in observer_list:
            result_list.append(["O", observer_time])
        # We sort all the time events by time
        event_list = sorted(result_list, key=lambda x: x[1])

        # # for i in range(len(event_list)-500):
        # #     if event_list[i][0] == "O":
        # #         print(event_list[i])
        # for e in event_list:
        #     print(e)
        
        # * We start to take into account the events
        # Declaration of variables
        total_num_packs_queue = 0
        total_observer_idles = 0

        for i in range(len(event_list)):
            #event_type= event_list.pop(0)
            event_type = event_list[i][0]
            # Arrival
            if event_type == 'A':
                num_arrival += 1
            elif event_type == 'D':
                num_departed += 1
            else:
                num_observers += 1
                # We record the num of packets that are currently on the uque
                total_num_packs_queue += (num_arrival-num_departed)
                if num_arrival == num_departed:
                    total_observer_idles += 1

        # print("total_num_packs = "+ str(total_num_packs_queue))
        # print("idles total= " +  str(total_num_packs_queue))
        # print("Total observers= "+str(num_observers))
        # print("Total arrivals= "+str(num_arrival))
        # print("Total departures= "+str(num_departed))
        return total_num_packs_queue/num_observers, total_observer_idles/num_observers

        # Now we wil calculate the departure time

    def __generate_exp_distribution(self, lambda_param: int) -> list:
        """
        This method is in charge of generating exponential random variables
        @param lambda_param: An integer that contains the average number of packets arrived
        @param size: An integer that defines how many numbers we generate
        @return list: We return a list with the numbers generated
        """
        expected_mean = 1/lambda_param
        return -expected_mean*math.log(1-random.random())

    # def m_m_1_k_queue(self):
    #     pass
    #     # TODO: FUNCION QUEUE LLENA
    #     # TODO: CHECK SI ARRIVAL LOSS OR NOT

    # # def __add_event(self, type, timestamp):

    # #     self.__event_list.append((type, timestamp))


# def generate_graph_points(avg_packet_length, trans_rate, t):
#     step = 0.1
#     start = 0.25
#     end = 0.95

#     result = []

#     i = start
#     while i < end:
#         lambda_para = trans_rate * i / avg_packet_length

#         list_m_m_1 = a.m_m_1_queue(
#             avg_packet_length, trans_rate, lambda_para, t)

#         result.append([i, list_m_m_1[0]])
#         print(list_m_m_1[0])
#         i += step

#     for e in result:
#         print(e)


if __name__ == "__main__":
    a = Lab1()
    num_packets = 1_000
    lambda_par = 75
    trans_rate = 1_000_000
    avg_packet_length = 2_000
    T = 1000
    # a.question1(lambda_par)

    sol_T = a.m_m_1_queue(avg_packet_length, trans_rate, lambda_par, T)
    mul_T = [x * 0.05 for x in sol_T]
    dif_T = [[x - y for x, y in zip(sol_T, mul_T)], [x + y for x, y in zip(sol_T, mul_T)]]

    sol_2T = a.m_m_1_queue(avg_packet_length, trans_rate, lambda_par, 2*T)
    mul_2T = [x * 0.05 for x in sol_2T]
    dif_2T = [[x - y for x, y in zip(sol_2T, mul_2T)], [x + y for x, y in zip(sol_2T, mul_2T)]]

    sol_3T = a.m_m_1_queue(avg_packet_length, trans_rate, lambda_par, 3*T)
    mul_3T = [x * 0.05 for x in sol_3T]
    dif_3T = [[x - y for x, y in zip(sol_3T, mul_3T)], [x + y for x, y in zip(sol_3T, mul_3T)]]

    difT_2T = [abs(x - y) for x, y in zip(sol_T, sol_2T)]
    print("The difference between T and 2T is: " + str(difT_2T))
    print("The range of difference 5% for T is: " + str(dif_T))
    inside_first = False
    if(dif_T[0] <= abs(sol_T-sol_2T) <= dif_T[1]): 
        inside_first = True
    print("Does T-2T enter in the difference? " + inside_first)


    dif2T_3T = [abs(x - y) for x, y in zip(sol_2T, sol_3T)]
    print("The difference between 2T and 3T is: " + str(dif2T_3T))
    print("The range of difference 5% for 2T is: " + str(dif_2T))
    inside_first = False
    if(dif_2T[0] <= <= dif_2T[1]): 
        inside_first = True
    print("Does 2T-3T enter in the difference? " + inside_first)




    #generate_graph_points(avg_packet_length, trans_rate, 1000)
