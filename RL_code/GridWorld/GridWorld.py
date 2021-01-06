import numpy as np
import time
#Defining Grid

#defining action variables
up=0
down=1
left=2
right=3

#no. of states and no. of variables
noS=4*4
noA=4

S=range(noS)
print("S:", S)
reward =-1    #for every step

terminal_state = lambda s: s==0 or s==noS-1  #first and last state terminal
print("terminal_state:", terminal_state)

wall=[]



P=dict() #transition probabilities

for s in S:
    print("s:", s)
    P[s]=dict()
    
    print("terminal_state(",s,"):",terminal_state(s))
    if (terminal_state(s)):
        P[s][up]=(s,1.0,0.0)   # next_state, probability, reward
        P[s][down]=(s,1.0,0.0)
        P[s][right]=(s,1.0,0.0)
        P[s][left]=(s,1.0,0.0)
        print("P[s]:", P[s])

    else:

        next_s= s if(s<4) else s-4
        print("next_s 1:", next_s)

        if next_s in wall:
            P[s][up]=(s,1.0,-1000.0)
        else:
            P[s][up]=(next_s,1.0,reward)
        print("P[",s,"][up]:", P[s][up])

        next_s= s if(16-s<=4) else s+4
        print("next_s 2:", next_s)

        if next_s in wall:
            P[s][down]=(s,1.0,-1000.0)
        else:
            P[s][down]=(next_s,1.0,reward)
        print("P[",s,"][down]:", P[s][down])


        next_s= s if((s+1)%4==0) else s+1
        print("next_s 3:", next_s)

        if next_s in wall:
            P[s][right]=(s,1.0,-1000.0)
        else:
            P[s][right]=(next_s,1.0,reward)
        print("P[",s,"][right]:", P[s][right])


        next_s= s if(s%4==0) else s-1
        print("next_s 4:", next_s)

        if next_s in wall:
            P[s][left]=(s,1.0,-1000.0)
        else:
            P[s][left]=(next_s,1.0,reward)
        print("P[",s,"][left]:", P[s][left])

        
print('No. of states in grid: ', noS)
print('No. of action options in each state:', noA)
Action_Index=dict()
Action_Index[0]='up'
Action_Index[1]='down'
Action_Index[2]='left'
Action_Index[3]='right'
Action_Index[5]='terminal states (stay)'
Action_Index[7]='wall'

print('Index for actions:')
for k,v in Action_Index.items():
    print(k,":",v)




#Policy Evaluation

def policy_evaluation(P,policy,threshold,discount):
    value=np.zeros((noS,))
    #print("value:", value)
    while True:
        new_value=np.zeros((noS,))

        change=0
        for s in S:
            print("------------------------------------")
            print("s en Funcion policy_evaluation:", s)
            v=0
            
            for a,action_prob in enumerate(policy[s]):
                print("a: ", a, "action_prob:", action_prob)
                next_state,probability,reward=P[s][a]
                print("next_state: ", next_state, "probability:", probability, "reward:", reward)
                print("value[",next_state,"]:", value[next_state])
                temp=probability*action_prob*(reward+discount*value[next_state])
                v+=temp
                print("temp:", temp)
                print("v:", v)

            change=max(change,np.abs(v-value[s]))
            new_value[s]=v
            print("new_value[",s,"]:", new_value[s])
            print("change:", change)

        if change < threshold:
              break

        value=new_value

    return value    



#Testing policy evaluation 
print("------------------------------")
random_policy = np.ones([16, 4])/4
print("random_policy:", random_policy)

threshold = 0.0001
discount = 1.0
value = np.zeros((noS,))
print("value:", value)

print("P:", P)
random_policy_value=policy_evaluation(P,random_policy,threshold,discount)

print("wall:", wall)
random_policy_value[wall]=13
print('Value Function for policy: all actions equiprobable:')
print(random_policy_value.reshape(4,4))




#Policy Evaluation SIN PRINTs

def policy_evaluation2(P,policy,threshold,discount):
    value=np.zeros((noS,))
    #print("value:", value)
    while True:
        new_value=np.zeros((noS,))

        change=0
        for s in S:
            #print("------------------------------------")
            #print("s en Funcion policy_evaluation:", s)
            v=0
            
            for a,action_prob in enumerate(policy[s]):
                #print("a: ", a, "action_prob:", action_prob)
                next_state,probability,reward=P[s][a]
                #print("next_state: ", next_state, "probability:", probability, "reward:", reward)
                #print("value[",next_state,"]:", value[next_state])
                temp=probability*action_prob*(reward+discount*value[next_state])
                v+=temp
                #print("temp:", temp)
                #print("v:", v)

            change=max(change,np.abs(v-value[s]))
            new_value[s]=v
            #print("new_value[",s,"]:", new_value[s])
            #print("change:", change)

        if change < threshold:
              break

        value=new_value

    return value    



#Testing policy evaluation 
#print("------------------------------")
random_policy = np.ones([16, 4])/4
#print("random_policy:", random_policy)

threshold = 0.0001
discount = 1.0
value = np.zeros((noS,))
#print("value:", value)

#print("P:", P)
random_policy_value=policy_evaluation2(P,random_policy,threshold,discount)

#print("------------------------------")

#print("random_policy_value:", random_policy_value)

#print("wall:", wall)
random_policy_value[wall]=13
#print('Value Function for policy: all actions equiprobable:')
#print(random_policy_value.reshape(4,4))


#Policy Iteration

def policy_iteration(P,discount,threshold):
    #Initialisation
    value=np.zeros((noS,))
    policy=np.ones([16, 4])/4
    
    while True:
        #Policy evaluation
        value=policy_evaluation2(P,policy,threshold,discount)
        print("value en function policy_iteration:", value)

        new_value=np.zeros((noS,))
        new_policy=np.zeros([noS,4])
        
        #Policy Improvement
        policy_stable=True
        for s in S:
            print("s:", s)
            if s!=0 and s!=15:
                old_action=policy[s]
                print("old_action:", old_action)

                action_values = np.zeros(noA)
                for a in range(noA):   		# Iterating over all the actions     
                        next_state,probability,reward = P[s][a]
                        print("next_state: ", next_state, "probability:", probability, "reward:", reward)

                        action_values[a] += probability*(reward + discount*value[next_state])
                        print("action_values[", a, "]:", action_values[a])

                max_total = np.amax(action_values)   # taking the max reward value 
                print("max_total:", max_total)
                best_a = np.argmax(action_values)
                print("best_a:", best_a)

                new_policy[s][best_a]=1

                new_value[s]=max_total 
                print("new_value[",s,"]:", new_value[s])
                if (np.array_equal(old_action,new_policy[s])!=True):
                    policy_stable=False
                    
        value=new_value
        print("value al final del WHILE:", value)
        if policy_stable:
            value[wall]=13
            return new_policy,value
        else:
            policy=new_policy 




print("P:", P)


start=time.clock()
best_policy,corr_value=policy_iteration(P,discount,threshold)
end=time.clock()

print("best_policy:", best_policy)
print("\n")
print("corr_value:",corr_value)

show_best_policy=np.zeros(noS,)
for s,p_s in enumerate(best_policy):
    if terminal_state(s):
        show_best_policy[s]=5
    elif s in wall:
        show_best_policy[s]=7
    else:
        show_best_policy[s]=np.argmax(p_s)

    
print('-------------- Best policy with Policy Iteration is ------------')
print(show_best_policy.reshape(4,4))
print('--------------- Corresponding Value Function is -----------------')
print(corr_value.reshape(4,4))
print('--------------- Time taken --------------------------------------')
print(end-start)



#Value Iteration

def value_iteration(P,discount,threshold):
    #Initialisation
    value=np.zeros((noS,))
    
    while True:
        new_policy=np.zeros([noS,4])
        change=0
        for s in S:
            if s!=0 and s!=15:
                v=value[s]
                action_values = np.zeros(noA)
                for a in range(noA):   		# Iterating over all the actions     
                        next_state,probability,reward = P[s][a]
                        action_values[a] += probability*(reward + discount*value[next_state])
                max_total = np.amax(action_values)   # taking the max reward value 
                best_a = np.argmax(action_values)

                value[s]=max_total
                new_policy[s][best_a]=1

                change=max(change,np.abs(v-value[s]))
            
        if change < threshold:
              break
    
    value[wall]=13            
    return new_policy,value.reshape(4,4)

start=time.clock()
best_policy,corr_value=value_iteration(P,discount,threshold)
end=time.clock()

show_best_policy=np.zeros(noS,)
for s,p_s in enumerate(best_policy):
    if terminal_state(s):
        show_best_policy[s]=5
    elif s in wall:
        show_best_policy[s]=7
    else:
        show_best_policy[s]=np.argmax(p_s)
    
print('Best policy with Value Iteration is')
print(show_best_policy.reshape(4,4))
print('Corresponding Value Function is')
print(corr_value.reshape(4,4))
print('Time taken')
print(end-start)




print('Our Value Function:')
print(corr_value.reshape(4,4))