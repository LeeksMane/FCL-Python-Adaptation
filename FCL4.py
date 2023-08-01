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
global agent_memories           # Stores the files in the memory of agents in non-repeated permutation notation
global agent_memories_repeated  # Stores the files in the memory of agents in repeated permutation notation
global updated_agent_memories   # Stores the files in the memory for the next iteration (used for iterations)
global updated_agent_memories_repeated      # Stores the files in the memory for the next iteration in repeated permutation notation
global agent_requests           # Stores the requested files for each agent
global agent_requests_mask      # Used for indicating whether a previous request still stand 
                                # (1:previous request still stands, 0: previous request fullfilled)
global agent_prob_dis           # Stores the mean and the variance of file request probability distributions
global agent_actions            # Stores the actions (requested files) for each agent
global exp_flag_record          # Records the behaviour of exploration stages
global agent_requests_record    # Records the statistics abour agent requests
global sf_mx_temp               # Decides on the temperature to be used for the soft max transformations
global index_mapping_dic        # Contains the index maps between non-repeated and repeated permutations
global inv_index_mapping_dic    # Contains the index maps between repeated and non-repeated permutations
global agent_rewards            # Contains the rewards accumulated in one iteration
global reward_records           # Contains the summation of the awards gathered in some episodes
global reward_amounts           # Decides how much each action will be rewarded
global state_progression_dic    # A list of dictionaries containing the mapping between states and actions
global learning_rate            # Defines the learning rate used in the iterations
global discount_rate            # Defines the discount rate used in the iterations

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
    global agent_memories_repeated
    global updated_agent_memories
    global updated_agent_memories_repeated
    global agent_requests
    global agent_requests_mask
    global agent_prob_dis
    global agent_actions
    global sf_mx_temp 
    global index_mapping_dic   
    global inv_index_mapping_dic
    global agent_rewards
    global reward_amounts 
    global state_progression_dic
    global learning_rate
    global discount_rate
    
    number_of_agents = 3     # Setting the number of agents
    number_of_files = 4     # Setting the number of files to be shared
    memory_of_agents = 2    # Setting the greatest number of files that
                            # can be stored in each agents memory
    number_of_per = math.perm(number_of_files,memory_of_agents)   # Number of file combinations
    
    # Creating the index transformation dictionaries
    create_index_mapping()
    
    # Creating the state transformation mapping dictionary
    create_state_mapping()
    
    # Deciding on the amount of rewards to be given to certain actions
    reward_amounts = np.zeros(5,dtype=int)
    reward_amounts[0] = 3   # Amount of reward recieved for already having the action file
    reward_amounts[1] = 2   # Amount of reward recieved for receiving the action file
    reward_amounts[2] = 1   # Amount of reward received for sending the action file to another agent
    reward_amounts[3] = 20  # Amount of reward received for having the requested file
    reward_amounts[4] = 0   # Amount of reward received for NOT having the requested file
    
    # Decide on the max number of epochs
    max_episode = 1000
    # Decide on the max number of time steps in each epoch
    max_iteration = 5000
    
    # Defining the learning rate
    learning_rate = 0.8
    
    # Defining the discount rate
    discount_rate = 0.75
    
    # Decide on the soft max temperature
    sf_mx_temp = 0.1
    
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
    
    # Initializing the memory array
    agent_memories = np.zeros(number_of_agents,dtype=int)
    agent_memories_repeated = np.zeros(number_of_agents,dtype=int)
    updated_agent_memories = np.zeros(number_of_agents,dtype=int)
    updated_agent_memories_repeated = np.zeros(number_of_agents,dtype=int)
    
    # Initializing the request array
    agent_requests = np.zeros(number_of_agents,dtype=int)
    agent_requests_mask = np.zeros(number_of_agents,dtype=bool)
    
    # Initializing the action array
    agent_actions = np.zeros(number_of_agents,dtype=int)
    
    # Initilazing the rewards array
    agent_rewards = np.zeros(number_of_agents,dtype=int)
    
    # Initialing the mean and the variances of prob dist
    agent_prob_dis = np.zeros((number_of_agents,2))
    for i in range(0,number_of_agents):
        agent_prob_dis[i,:] = np.array([0.2+((0.6/(number_of_agents-1))*i),0.1])

    exp_flag = 0
    episode_number = 0
    iteration_number = 0

# This function creates the dictionary mapping the memory 
# indexes in non-repeated permutation form to repeated 
# permutation form 
def create_index_mapping():
    global number_of_agents
    global number_of_files
    global number_of_per
    global index_mapping_dic   
    global inv_index_mapping_dic
    
    index_mapping_dic = {}
    inv_index_mapping_dic = {}
    
    index1 = 0
    index2 = 0
    
    for i in range(0,number_of_files):
        for j in range(0,number_of_files):
            # If i != j, then there is an entry on the non-repeated permutations index
            if i != j:
                index_mapping_dic.update({index1:index2})
                inv_index_mapping_dic.update({index2:index1})
            if i == j:
                index1 += 0     # When i==j, no entry is added to the non-repeated permutations
                index2 += 1     # When i==j, an entry is added to the repeated permutations
            else:
                index1 += 1     # When i!=j, an entry is added to the non-repeated permutations
                index2 += 1     # When i!=j, an entry is added to the repeated permutation
                 
# This function creates the dictionary mapping previous 
# state tp the new states using the actions as a guide
def create_state_mapping():
    global number_of_per
    global number_of_files
    global state_progression_dic
    
    state_progression_dic = []
    for prev_state in range(0,number_of_per):                   # Going through each possible state
        state_progression_dic.append({})
        prev_state_rep = index_mapping_dic[prev_state]          # Converting to repeating permutation 
        for act in range(0,number_of_files):                    # Going through each possible action
            new_state_rep = (act*number_of_files)+(prev_state_rep // number_of_files)
            # If the new state is not in this dictionary, then the action was 
            # illegal (a file that was already present was requested)
            if new_state_rep in inv_index_mapping_dic:          
                new_state = inv_index_mapping_dic[new_state_rep]    # Converting to non-repeating permutation
                state_progression_dic[prev_state].update({act:new_state})
                  
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
    global agent_requests_mask
    global agent_prob_dis
    
    for i in range(0,number_of_agents):
        if agent_requests_mask[i] == 0:     
            # If the mask contains 1, then the previous request is yet to be 
            # fullfilled then thus there is no need to generate a new request
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
        if (episode_number == (max_episode-1)) and (iteration_number == (max_iteration-1)):
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
    
# This function converts the agent memories from the non-repeated 
# permutations notation to the repeated premutations notation
def convert_memories(updated_conversion,inverse_conversion):
    global agent_memories
    global agent_memories_repeated
    global updated_agent_memories
    global updated_agent_memories_repeated
    global index_mapping_dic
    global inv_index_mapping_dic
    
    # If the updated_conversion is selected the transformation 
    # occurs between the updated memories
    
    # If the inverse_conversion is selected then the notation 
    # is converted from the repeated permutation to non-repeated
    # permutation
    if inverse_conversion == True:
        if updated_agent_memories == False:
            agent_memories = np.vectorize(inv_index_mapping_dic.get)(agent_memories_repeated)
        elif updated_agent_memories == True:
            updated_agent_memories = np.vectorize(inv_index_mapping_dic.get)(updated_agent_memories_repeated)
    # If the inverse_conversion is not selected then the notation 
    # is converted from the non-repeated permutation to repeated
    # permutation
    elif inverse_conversion == False:
        if updated_conversion == False:
            agent_memories_repeated = np.vectorize(index_mapping_dic.get)(agent_memories)
        elif updated_conversion == True:
            updated_agent_memories_repeated = np.vectorize(index_mapping_dic.get)(updated_agent_memories)
            
# This function updates the states (files in the memory) 
# based on the actions of the agents (requested files)  
def apply_actions():
    global agent_actions
    global agent_memories
    global agent_memories_repeated
    global updated_agent_memories 
    global updated_agent_memories_repeated 
    global index_mapping_dic   
    global inv_index_mapping_dic
    global agent_rewards
    global reward_amounts
    global state_progression_dic
    
    # Converting the agent memories to repeated permutaion notation
    convert_memories(updated_conversion=False,inverse_conversion=False)
    for i in range(0,number_of_agents):
        act_fil = agent_actions[i]      # Extracting the action file for the agent
        # If this condition is fullfilled, the action file is in the memory of the agent already
        if (agent_memories_repeated[i]//number_of_files == act_fil) or (agent_memories_repeated[i]%number_of_files == act_fil):
            agent_rewards[i] += reward_amounts[0]
            updated_agent_memories [i] = agent_memories[i]
        else:   # The action file is not present at the agent already
            for j in range(0,number_of_agents):     # Lookin at the memories of other agents to find the action file 
                # If this condition is fullfilled, the action file is in the memory of another agent:
                if (agent_memories_repeated[j]//number_of_files == act_fil) or (agent_memories_repeated[j]%number_of_files == act_fil):
                    agent_rewards[i] += reward_amounts[1]
                    agent_rewards[j] += reward_amounts[2]
                    updated_agent_memories [i] = state_progression_dic[agent_memories[i]][act_fil]
                    break
# This function checks whether the requests were fullfilled 
# based on the updated states and modifies the request mask
# and distributes adequte rewards
def check_requests(): 
    global agent_requests
    global agent_requests_mask
    global agent_memories
    global agent_memories_repeated
    global updated_agent_memories 
    global updated_agent_memories_repeated 
    global index_mapping_dic   
    global inv_index_mapping_dic
    global agent_rewards
    global reward_amounts
    
    # Converting the agent memories to repeated permutaion notation
    convert_memories(updated_conversion=True,inverse_conversion=False)

    for i in range(0,number_of_agents):
        req_fil = agent_requests[i]
        # If this condition is fullfilled, the requested file is in the memory of the agent already:
        if (updated_agent_memories_repeated[i]//number_of_files == req_fil) or (updated_agent_memories_repeated[i]%number_of_files == req_fil):
            agent_rewards[i] += reward_amounts[3]
            agent_requests_mask[i] = 0
        # If not, then the request is yet to be fullfilled:  
        else:
            agent_rewards[i] += reward_amounts[4]
            agent_requests_mask[i] = 1
            
# This function records and displays the sum of collective rewards for all epochs           
def record_rewards(record_flag,record_frequency):
    global episode_number
    global iteration_number
    global max_episode
    global max_iteration
    global agent_rewards
    global reward_records
    
    if record_flag == True:
        if (episode_number == 0) and (iteration_number==0):
            # Creating the record array for the cumulative rewards
            reward_records = np.zeros(((max_episode//record_frequency),max_iteration+1))
        if (episode_number % record_frequency) == 0:    
            # Recording the cumulative rewards
            reward_records[int(episode_number//record_frequency),
            int(iteration_number+1)] = reward_records[int(episode_number//record_frequency),
            int(iteration_number)] + sum(agent_rewards)
        if (episode_number == (max_episode-1)) and (iteration_number == (max_iteration-1)):
            means = np.add.reduce(reward_records,axis=0)
            means /= (max_episode//record_frequency)
            reward_records[:,:] -= means
            # At the end of the recording process, the plots are created   
            for i in range(0,(max_episode//record_frequency)):
                if i < (max_episode//record_frequency)-10:
                    plt.plot(np.arange(0,max_iteration,1),reward_records[i,1:max_iteration+1],
                            color=(max(0,0.25-0.005*i),max(0,0.35-0.005*i),min(0.50+0.004*i,1),min(0.15+0.015*i,1)) )
                else:
                    plt.plot(np.arange(0,max_iteration,1),reward_records[i,1:max_iteration+1],
                            color=(min(1,0.8+0.01*i),0,0,1) )
                plt.title("Cumulative Rewards Through the Episodes:", 
                        y=0.8,
                        fontsize = 8)
                plt.xlabel("Iteration Number") 
                plt.ylabel("Cumulative Rewards")
            plt.show()    
       
# This function updates the QC values using the rewards gathered by all agents
def update_QC_values():
    global QC
    global agent_memories
    global updated_agent_memories
    global agent_actions
    global agent_rewards
    global learning_rate
    global discount_rate
    
    # Updating the QC values using the collective rewards
    QC[tuple(agent_memories), 
    tuple(agent_actions)] = (1-learning_rate)*QC[tuple(agent_memories),
    tuple(agent_actions)] + learning_rate*sum(agent_rewards) + learning_rate*discount_rate*np.max(QC[tuple(updated_agent_memories)])
       
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
        
        # Clearing the agent request mask
        agent_requests_mask = np.zeros(number_of_agents)
        
        for iteration_number in range(0,max_iteration):
           
            # Deciding on the exploration
            decide_exploration()
            
            # Recording exploration stages
            # record_flag == True -> Record and display
            record_exp_flag(record_flag=False, recorded_episodes=[1,100,200,399])
            
            # Deciding on the actions based on the exploration 
            # flag and the Q function values
            decide_actions()
            
            # Modifying the states (files in the memory) based 
            # on the actions (requested files)
            apply_actions()
            
            # Generating requests for each agent
            generate_requests()
            
            # Recording agent requests
            # record_flag == True -> Record and display
            record_requests(record_flag=False)
            
            # Checking whether the requested were 
            # fullfilled at this iteration
            check_requests()
            
            # Recording the rewards throughout the epochs
            # record_flag == True -> Record and display
            # One in record_frequeny episodes will be recorded
            record_rewards(record_flag=True,record_frequency=25)
            
            # Updating the QC values using the rewards gathered
            update_QC_values()
            
            # Clearing the rewards
            agent_rewards[:] = 0
            
            # Since the iteration is complete, the updated 
            # state can be written to the memories array for 
            # itearation to continue
            agent_memories = updated_agent_memories
            
        print("The episode number "+str(episode_number)+" is complete...")        
            
            