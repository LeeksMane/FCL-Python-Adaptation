#
#
# Vehicle to Vehicle Transmission 
# Coorditaned by the FCL Algortihm 
#
# Author: Kemal Enes Akyüz
#

import numpy as np
import random
import math
import matplotlib.pyplot as plt

# Defining the global variables
global number_of_agents         # Number of agents in the scenario
global number_of_files          # Number of files that can be exchanged between agents
global memory_of_agents         # Maximum number of files that can be stored in an agent's memory
global number_of_per            # Number of possible file permutations on an agent's memory
global exp_flag                 # Dictates whether exploration will occur at a step
global episode_number           # Current number of the epoch
global max_episode              # Number of epochs to be completed
global iteration_number         # Current value of time
global max_iteration            # Number of time steps to be completed in each epoch
global QC                       # Cooperative Q function Matrix
global agent_memories           # Stores the two files in the memory of the agents
global agent_requests           # Stores the requested files for each agent
global agent_prob_dis           # Stores the mean and the variance of file request probability distributions
global agent_actions            # Stores the actions (requested files) for each agent
global exp_flag_record          # Records the behaviour of exploration stages
global agent_requests_record    # Records the statistics abour agent requests
global sf_mx_temp               # Decides on the temperature to be used for the soft max transformations
global agent_expectations       # Stores the expected Q value for different actions given the state

# This function initiliazes the global variables 
# and creates appropriate sized arrays and 
# matrices for all necessary variables
def init_global_variables():
    global number_of_agents
    global number_of_files
    global memory_of_agents
    global number_of_per
    global exp_flag
    global episode_number
    global max_episode
    global iteration_number
    global max_iteration
    global QC
    global agent_memories
    global agent_requests
    global agent_prob_dis
    global agent_actions
    global sf_mx_temp 
    global agent_expectations
    
    number_of_agents = 3     # Setting the number of agents
    number_of_files = 4     # Setting the number of files to be shared
    memory_of_agents = 2    # Setting the greatest number of files that
                            # can be stored in each agents memory
    number_of_per = math.perm(number_of_files,memory_of_agents)   # Number of file combinations
    
    # Decide on the max number of epochs
    max_episode = 400
    # Decide on the max number of time steps in each epoch
    max_iteration = 1000
    
    # Decide on the soft max temperature
    sf_mx_temp = 1
    
    # Generating the cooperative utility matrix
    temp_list = []
    for i in range(0,2):
        for j in range(0,number_of_agents):
            if i == 0:
                temp_list.append(number_of_per)
            else:
                temp_list.append(number_of_files)
    if ((number_of_per**number_of_agents)*(number_of_files**number_of_agents)) < 200000:
        QC = np.zeros(temp_list)
    else:
        print("Q array size too large!(" + str(((number_of_per**number_of_agents)*(number_of_files**number_of_agents))) + ")")    
    
    # The Cooperative Q function Matrix QC is indexed first by the 
    # states of the files in the memoris of each agent. For each 
    # agent there are perm(number_of_files,memory_of_agents) 
    # different states possible since the order of the files on 
    # the memory also matters. The remaining indices relate to the 
    # action space and express the number of the requested file for 
    # each agent: 
    # QC[state of agent 1, ..., state of agent N, action of agent 1, ..., action of agent N]
    
    # Initializing the array for the expectations for actions given a state
    agent_expectations = np.zeros((number_of_agents,number_of_files))
    
    # Initializing the memory array
    agent_memories = np.zeros(number_of_agents,dtype=int)

    # Initializing the request array
    agent_requests = np.zeros(number_of_agents,dtype=int)
    
    # Initializing the action array
    agent_actions = np.zeros(number_of_agents,dtype=int)
    
    # Initialing the mean and the variances of prob dist
    agent_prob_dis = np.zeros((number_of_agents,2))
    for i in range(0,number_of_agents):
        agent_prob_dis[i,:] = np.array([0.2+((0.6/(number_of_agents-1))*i),0.1])

    exp_flag = 0
    episode_number = 0
    iteration_number = 0

# This function can be used to display the shape of an array
def display_array_shape(display_flag, array):
    if display_flag == True:
        print(np.shape(array))
        print(np.size(array))   

# This function decides whether an
# exploration stage will occur based 
# on episode and iteration numbers
def decide_exploration():
    global episode_number
    global iteration_number
    global exp_flag
    
    # Generating a pseudo-random number between 0 and 1
    r = random.uniform(0,1)
    # Calculating the probability of exploration for a given
    # epoch number and time
    exp_cutoff = math.exp(-0.0053*(1+episode_number*0.05)*iteration_number)
    if r < exp_cutoff:
        exp_flag = 1
    else: 
        exp_flag = 0

# This funciton randomly assigns two files to each agents' memory
def init_files_in_memory():
    global number_of_agents
    global number_of_per
    global agent_memories
    
    for i in range(0,number_of_agents):
        agent_memories[i] = random.randint(0,number_of_per-1)

# This function generates random file request for each agent
def generate_requests():
    global number_of_agents
    global number_of_files
    global agent_requests
    global agent_prob_dis
    
    for i in range(0,number_of_agents):
        r = 2
        while ((r<=0) or (r>=1)): 
            r = np.random.normal(agent_prob_dis[i,0],agent_prob_dis[i,1])
        agent_requests[i] = (r//(1/number_of_files))
    # "0": 0th file requested, "1": 1st file requested, ...

# This function records the behaviour of the exploration flag 
# through the episodes
def record_exp_flag(record_flag, recorded_episodes):
    global exp_flag
    global episode_number
    global max_episode
    global iteration_number
    global max_iteration
    global exp_flag_record
    
    if record_flag == True:     # If a record is requested
        if episode_number == 0:
            exp_flag_record = np.zeros((len(recorded_episodes),max_iteration),dtype=bool)    # Creating a record array
        if episode_number in recorded_episodes:
            # Recording the exploration flag data
            exp_flag_record[recorded_episodes.index(episode_number),iteration_number] = exp_flag
        if episode_number == (max_episode-1):
            # At the end of the recording process, the plots are created
            for i in range(0,len(recorded_episodes)):
                plt.subplot(len(recorded_episodes),1,i+1)
                plt.plot(np.arange(0,max_iteration,1),exp_flag_record[i,:])
                plt.title("Exploration Flag at Episode: " + str(recorded_episodes[i]),
                        loc='right', 
                        y=0.8,
                        fontsize = 8)
                plt.xlabel('Iteration Number') 
                plt.ylabel('Exploration Flag') 
            plt.show()

# This function records and displays the requests of each agent
def record_requests(record_flag):
    global episode_number
    global agent_requests
    global agent_requests_record
    global number_of_agents
    global number_of_files
    global max_episode
    global max_iteration
    
    if record_flag == True:
        if episode_number == 0:
            agent_requests_record = np.zeros((number_of_agents,number_of_files))
        for i in range(0,number_of_agents):
            agent_requests_record[i,int(agent_requests[i])] += 1
        if episode_number == (max_episode-1):
            for i in range(0,number_of_agents):
                plt.subplot((number_of_agents//2)+1,2,i+1)
                plt.stem(np.arange(0,number_of_files,1),
                        agent_requests_record[i,:]/(max_episode*max_iteration))
                plt.title("File Requests by Agent: " + str(i),
                                        loc='right', 
                                        y=0.8,
                                        fontsize = 8)
                plt.xlabel('File Number') 
                plt.ylabel('Ratio of Requests') 
            plt.show()
            
# This function decides on the actions of agents based on 
# the exploration flag and the QC matrix
def decide_actions():
    global exp_flag
    global agent_actions
    global QC
    global number_of_agents
    global number_of_files
    global agent_memories
    global agent_expectations
    
    # If there is an exploration stage, actions are 
    # randomly selected with uniform probability
    if exp_flag == 1:   
        for i in range(0,number_of_agents):
            agent_actions[i] = random.randint(0,number_of_files-1)

    # If there is no exploration stage, actions are 
    # determined by a softmax transformation
    else:
        # Getting all the possible Q values for all 
        # the actions based on the currrent state
        QC_RED = QC[tuple(agent_memories)]
        
        for i in range(0,number_of_agents):
            # Finding the average Q values for each action 
            # for a given agent
            
            # Creating the tuple to be used for indexing the summation
            tp = tuple(np.delete(np.arange(0,number_of_agents),i))  
            # Summing all the Q values for a given action
            average_q_values = np.add.reduce(QC_RED,axis=tp)
            # Apllying the softmax transformation
            pd = soft_max_transformer(average_q_values)
            
            # Producing a probability distribution depending on the 
            # soft max of the average Q value a action has for a given state
            agent_actions[i] = np.random.choice(np.arange(0, number_of_files), p=pd)
            
               
# This function converts a set of values in an array through 
# a soft max transformation
def soft_max_transformer(input_array):
    global sf_mx_temp
    
    soft_max_array = np.zeros(len(input_array))
    for i in range(0,len(input_array)):
        soft_max_array[i] = math.exp(sf_mx_temp*input_array[i])
    
    return np.half(soft_max_array/sum(soft_max_array))
    
    
            
if __name__ == "__main__":
    
    # Setting the print options for the numpy package
    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)
    
    # Initializing the global variables
    init_global_variables()
    
    # Displaying the shape and size of the QC matrix
    display_array_shape(display_flag=False, array=QC)

    for episode_number in range(0,max_episode):
        # Distributing random files to agents initially
        init_files_in_memory()
        for iteration_number in range(0,max_iteration):
           
            # Deciding on the exploration
            decide_exploration()
            
            # Recording exploration stages
            # record_flag == True -> Record and display
            record_exp_flag(record_flag=False, recorded_episodes=[1,100,200,399])
            
            # Generating requests for each agent
            generate_requests()
            
            # Recording agent requests
            # record_flag == True -> Record and display
            record_requests(record_flag=False)
            
            # Deciding on the actions based on the exploration 
            # flag and the Q function values
            decide_actions()
            
            
            
            
            
            
            


