import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
import sys

grid_val = np.zeros((10,10))

transition_N = np.zeros((100,100))

imp_dic = {}
cnt = 0
for row in range(10):
    for col in range(10):
        key = (row, col)
        imp_dic[key] = cnt
        cnt+=1


#np.fill_diagonal(transition_N, 0.1) #filling diagonal for prob of staying
                                    #at same state

for r in range(10):
    for c in range(10):
        #should be able to handle changes in r and c hence all middle parts
        if(r>0 and c>0 and r<9 and c<9):
            intr_r = r
            intr_c = c

            row_in_t = imp_dic[(intr_r, intr_c)] 
            #get the row of t matrix and then only cols will change

            #now getting the 4 cols where I need to put in the transitions
            col1 = imp_dic[(intr_r, intr_c)]
            col2 = imp_dic[(intr_r-1, intr_c)]
            col3 = imp_dic[(intr_r, intr_c-1)]
            col4 = imp_dic[(intr_r, intr_c+1)]

            transition_N[row_in_t, col1] = 0.1
            transition_N[row_in_t, col2] = 0.7
            transition_N[row_in_t, col3] = 0.1
            transition_N[row_in_t, col4] = 0.1

        if(r==0 and c==0): #if first row first col and go north. 
            #Assign staying at same and going east same prob

            get_r = imp_dic[(r, c)]

            get_c1 = imp_dic[(r,c)]
            get_c2 = imp_dic[(r, c+1)]

            transition_N[get_r, get_c1] = 0.5 
            transition_N[get_r, get_c2] = 0.5
        
        if(r == 0 and c==9): #if first row last col and go north. 
            f_r = imp_dic[(r, c)]

            f_c1 = imp_dic[(r,c)]
            f_c2 = imp_dic[(r, c-1)]

            transition_N[f_r, f_c1] = 0.5 
            transition_N[f_r, f_c2] = 0.5

        if(r==0 and c>0 and c<9): #entire first row leaving edges
            a_r = imp_dic[(r, c)]

            a_c1 = imp_dic[(r, c)]
            a_c2 = imp_dic[(r, c-1)]
            a_c3 = imp_dic[(r, c+1)]

            transition_N[a_r, a_c1] = 0.33
            transition_N[a_r, a_c2] = 0.33
            transition_N[a_r, a_c3] = 0.33

        if(r>0 and c==0): #leaving first cell and taking entire col
            b_r = imp_dic[(r, c)]

            b_c1 = imp_dic[(r, c)]
            b_c2 = imp_dic[(r-1, c)]
            b_c3 = imp_dic[(r, c+1)]

            transition_N[b_r, b_c1] = 0.2 #staying at same place coz wall
            transition_N[b_r, b_c2] = 0.7
            transition_N[b_r, b_c3] = 0.1

        if(r>0 and c==9): 

            c_r = imp_dic[(r, c)]

            c_c1 = imp_dic[(r, c)]
            c_c2 = imp_dic[(r-1, c)]
            c_c3 = imp_dic[(r, c-1)]

            transition_N[c_r, c_c1] = 0.2 #staying at same place coz wall
            transition_N[c_r, c_c2] = 0.7
            transition_N[c_r, c_c3] = 0.1
        
        if(r==9 and c>0 and c<9): #lat row leaving edges
            d_r = imp_dic[(r,c)]

            d_c1 = imp_dic[(r,c)]
            d_c2 = imp_dic[(r-1,c)]
            d_c3 = imp_dic[(r,c-1)]
            d_c4 = imp_dic[(r,c+1)]

            transition_N[d_r, d_c1] = 0.1
            transition_N[d_r, d_c2] = 0.7
            transition_N[d_r, d_c3] = 0.1
            transition_N[d_r, d_c4] = 0.1



transition_S = np.zeros((100,100))


for r in range(10):
    for c in range(10):
        #should be able to handle changes in r and c hence all middle parts
        if(r>0 and c>0 and r<9 and c<9):
            intr_r = r
            intr_c = c

            row_in_t = imp_dic[(intr_r, intr_c)] 
            #get the row of t matrix and then only cols will change

            #now getting the 4 cols where I need to put in the transitions
            col1 = imp_dic[(intr_r, intr_c)]
            col2 = imp_dic[(intr_r+1, intr_c)]
            col3 = imp_dic[(intr_r, intr_c-1)]
            col4 = imp_dic[(intr_r, intr_c+1)]

            transition_S[row_in_t, col1] = 0.1
            transition_S[row_in_t, col2] = 0.7
            transition_S[row_in_t, col3] = 0.1
            transition_S[row_in_t, col4] = 0.1

        if(r==9 and c==0): #if last row first col and go south. 
            #Assign staying at same and going east same prob

            get_r = imp_dic[(r, c)]

            get_c1 = imp_dic[(r,c)]
            get_c2 = imp_dic[(r, c+1)]

            transition_S[get_r, get_c1] = 0.5 
            transition_S[get_r, get_c2] = 0.5
        
        if(r == 9 and c==9): #if last row last col and go south. 
            f_r = imp_dic[(r, c)]

            f_c1 = imp_dic[(r,c)]
            f_c2 = imp_dic[(r, c-1)]

            transition_S[f_r, f_c1] = 0.5 
            transition_S[f_r, f_c2] = 0.5

        if(r==9 and c>0 and c<9): #entire last row leaving edges
            a_r = imp_dic[(r, c)]

            a_c1 = imp_dic[(r, c)]
            a_c2 = imp_dic[(r, c-1)]
            a_c3 = imp_dic[(r, c+1)]

            transition_S[a_r, a_c1] = 0.33
            transition_S[a_r, a_c2] = 0.33
            transition_S[a_r, a_c3] = 0.33

        if(r<9 and c==0): #leaving last cell and taking entire col
            b_r = imp_dic[(r, c)]

            b_c1 = imp_dic[(r, c)]
            b_c2 = imp_dic[(r+1, c)]
            b_c3 = imp_dic[(r, c+1)]

            transition_S[b_r, b_c1] = 0.2 #staying at same place coz wall
            transition_S[b_r, b_c2] = 0.7
            transition_S[b_r, b_c3] = 0.1

        if(r<9 and c==9): 

            c_r = imp_dic[(r, c)]

            c_c1 = imp_dic[(r, c)]
            c_c2 = imp_dic[(r+1, c)]
            c_c3 = imp_dic[(r, c-1)]

            transition_S[c_r, c_c1] = 0.2 #staying at same place coz wall
            transition_S[c_r, c_c2] = 0.7
            transition_S[c_r, c_c3] = 0.1
        
        if(r==0 and c>0 and c<9): #last row leaving edges
            d_r = imp_dic[(r,c)]

            d_c1 = imp_dic[(r,c)]
            d_c2 = imp_dic[(r+1,c)]
            d_c3 = imp_dic[(r,c-1)]
            d_c4 = imp_dic[(r,c+1)]

            transition_S[d_r, d_c1] = 0.1
            transition_S[d_r, d_c2] = 0.7
            transition_S[d_r, d_c3] = 0.1
            transition_S[d_r, d_c4] = 0.1



transition_W = np.zeros((100,100))


for r in range(10):
    for c in range(10):
        #should be able to handle changes in r and c hence all middle parts
        if(r>0 and c>0 and r<9 and c<9):
            intr_r = r
            intr_c = c

            row_in_t = imp_dic[(intr_r, intr_c)] 
            #get the row of t matrix and then only cols will change

            #now getting the 4 cols where I need to put in the transitions
            col1 = imp_dic[(intr_r, intr_c)]
            col2 = imp_dic[(intr_r, intr_c-1)]
            col3 = imp_dic[(intr_r-1, intr_c)]
            col4 = imp_dic[(intr_r+1, intr_c)]

            transition_W[row_in_t, col1] = 0.1
            transition_W[row_in_t, col2] = 0.7
            transition_W[row_in_t, col3] = 0.1
            transition_W[row_in_t, col4] = 0.1

        if(r==0 and c==0): #if first row first col and go west. 
            #Assign staying at same and going east same prob

            get_r = imp_dic[(r, c)]

            get_c1 = imp_dic[(r,c)]
            get_c2 = imp_dic[(r+1, c)]

            transition_W[get_r, get_c1] = 0.5 
            transition_W[get_r, get_c2] = 0.5
        
        if(r == 9 and c==0): #if last row first col and go west. 
            f_r = imp_dic[(r, c)]

            f_c1 = imp_dic[(r,c)]
            f_c2 = imp_dic[(r-1, c)]

            transition_W[f_r, f_c1] = 0.5 
            transition_W[f_r, f_c2] = 0.5

        if(c==0 and r>0 and r<9): #entire first column leaving edges
            a_r = imp_dic[(r, c)]

            a_c1 = imp_dic[(r, c)]
            a_c2 = imp_dic[(r+1, c)]
            a_c3 = imp_dic[(r-1, c)]

            transition_W[a_r, a_c1] = 0.33
            transition_W[a_r, a_c2] = 0.33
            transition_W[a_r, a_c3] = 0.33

        if(c>0 and r==0): #leaving last cell and taking entire col
            b_r = imp_dic[(r, c)]

            b_c1 = imp_dic[(r, c)]
            b_c2 = imp_dic[(r, c-1)]
            b_c3 = imp_dic[(r+1, c)]

            transition_W[b_r, b_c1] = 0.2 #staying at same place coz wall
            transition_W[b_r, b_c2] = 0.7
            transition_W[b_r, b_c3] = 0.1

        if(c>0 and r==9): 

            c_r = imp_dic[(r, c)]

            c_c1 = imp_dic[(r, c)]
            c_c2 = imp_dic[(r, c-1)]
            c_c3 = imp_dic[(r-1, c)]

            transition_W[c_r, c_c1] = 0.2 #staying at same place coz wall
            transition_W[c_r, c_c2] = 0.7
            transition_W[c_r, c_c3] = 0.1
        
        if(c==9 and r>0 and r<9): #entire last column leaving edges
            d_r = imp_dic[(r,c)]

            d_c1 = imp_dic[(r,c)]
            d_c2 = imp_dic[(r,c-1)]
            d_c3 = imp_dic[(r-1,c)]
            d_c4 = imp_dic[(r+1,c)]

            transition_W[d_r, d_c1] = 0.1
            transition_W[d_r, d_c2] = 0.7
            transition_W[d_r, d_c3] = 0.1
            transition_W[d_r, d_c4] = 0.1


transition_E = np.zeros((100,100))


for r in range(10):
    for c in range(10):
        #should be able to handle changes in r and c hence all middle parts
        if(r>0 and c>0 and r<9 and c<9):
            intr_r = r
            intr_c = c

            row_in_t = imp_dic[(intr_r, intr_c)] 
            #get the row of t matrix and then only cols will change

            #now getting the 4 cols where I need to put in the transitions
            col1 = imp_dic[(intr_r, intr_c)]
            col2 = imp_dic[(intr_r, intr_c+1)]
            col3 = imp_dic[(intr_r-1, intr_c)]
            col4 = imp_dic[(intr_r+1, intr_c)]

            transition_E[row_in_t, col1] = 0.1
            transition_E[row_in_t, col2] = 0.7
            transition_E[row_in_t, col3] = 0.1
            transition_E[row_in_t, col4] = 0.1

        if(r==0 and c==9): #if first row last col and go east. 
            #Assign staying at same and going south same prob

            get_r = imp_dic[(r, c)]

            get_c1 = imp_dic[(r,c)]
            get_c2 = imp_dic[(r+1, c)]

            transition_E[get_r, get_c1] = 0.5 
            transition_E[get_r, get_c2] = 0.5
        
        if(r == 9 and c==9): #if last row first col and go west. 
            f_r = imp_dic[(r, c)]

            f_c1 = imp_dic[(r,c)]
            f_c2 = imp_dic[(r-1, c)]

            transition_E[f_r, f_c1] = 0.5 
            transition_E[f_r, f_c2] = 0.5

        if(c==9 and r>0 and r<9): #entire first column leaving edges
            a_r = imp_dic[(r, c)]

            a_c1 = imp_dic[(r, c)]
            a_c2 = imp_dic[(r+1, c)]
            a_c3 = imp_dic[(r-1, c)]

            transition_E[a_r, a_c1] = 0.33
            transition_E[a_r, a_c2] = 0.33
            transition_E[a_r, a_c3] = 0.33

        if(c<9 and r==0): #leaving last cell and taking entire col
            b_r = imp_dic[(r, c)]

            b_c1 = imp_dic[(r, c)]
            b_c2 = imp_dic[(r, c+1)]
            b_c3 = imp_dic[(r+1, c)]

            transition_E[b_r, b_c1] = 0.2 #staying at same place coz wall
            transition_E[b_r, b_c2] = 0.7
            transition_E[b_r, b_c3] = 0.1

        if(c<9 and r==9): 

            c_r = imp_dic[(r, c)]

            c_c1 = imp_dic[(r, c)]
            c_c2 = imp_dic[(r, c+1)]
            c_c3 = imp_dic[(r-1, c)]

            transition_E[c_r, c_c1] = 0.2 #staying at same place coz wall
            transition_E[c_r, c_c2] = 0.7
            transition_E[c_r, c_c3] = 0.1
        
        if(c==0 and r>0 and r<9): #entire last column leaving edges
            d_r = imp_dic[(r,c)]

            d_c1 = imp_dic[(r,c)]
            d_c2 = imp_dic[(r,c+1)]
            d_c3 = imp_dic[(r-1,c)]
            d_c4 = imp_dic[(r+1,c)]

            transition_E[d_r, d_c1] = 0.1
            transition_E[d_r, d_c2] = 0.7
            transition_E[d_r, d_c3] = 0.1
            transition_E[d_r, d_c4] = 0.1


row = imp_dic[(8, 8)] #adding goal transition
transition_E[row,:] = 0
transition_E[row,row] = 1

transition_N[row,:] = 0
transition_N[row,row] = 1

transition_S[row,:] = 0
transition_S[row,row] = 1

transition_W[row,:] = 0
transition_W[row,row] = 1

cost_mat = np.zeros((10,10))
value_mat = np.zeros((10,10))

cost_mat += -1 #reward of every move before adding obs and terminal state

cost_mat[7, 3:7] = -10 #bich me obs
cost_mat[2:6, 4] = -10
cost_mat[2,5] = -10
cost_mat[4:6,7] = -10

cost_mat[:, 0] = -10
cost_mat[:, 9] = -10
cost_mat[0, :] = -10
cost_mat[9, :] = -10

cost_mat[8, 8] = 10
# print(cost_mat)
# sys.exit()
gamma = 0.9
diff = 15
val_old = deepcopy(value_mat)
while(diff>0.000005): #policy evaluation
    value_mat = cost_mat + gamma*(np.reshape(transition_E@value_mat.flatten(), (10,10)))
    diff = np.linalg.norm(value_mat- val_old)
    #print(diff)
    val_old = value_mat



#plt.imshow(value_mat)
# img = sns.heatmap(value_mat,annot=True)
# plt.show()
# sys.exit()

#policy improvement

# value_mat_N = cost_mat + gamma*(np.reshape(transition_N@value_mat.flatten(), (10,10)))
# value_mat_S = cost_mat + gamma*(np.reshape(transition_S@value_mat.flatten(), (10,10)))
# value_mat_E = cost_mat + gamma*(np.reshape(transition_E@value_mat.flatten(), (10,10)))
# value_mat_W = cost_mat + gamma*(np.reshape(transition_W@value_mat.flatten(), (10,10)))

action_mat = np.zeros((10,10))
action_name_mat = np.zeros((10,10), dtype= str)
new_transition = np.zeros((100,100))
act_name = np.array(["N", "S", "E", "W"], dtype = str)

for fin in range(4):


    value_mat_N = cost_mat + gamma*(np.reshape(transition_N@value_mat.flatten(), (10,10)))
    value_mat_S = cost_mat + gamma*(np.reshape(transition_S@value_mat.flatten(), (10,10)))
    value_mat_E = cost_mat + gamma*(np.reshape(transition_E@value_mat.flatten(), (10,10)))
    value_mat_W = cost_mat + gamma*(np.reshape(transition_W@value_mat.flatten(), (10,10)))

    for r in range(10):
        for c in range(10):
            mat = np.array([value_mat_N[r,c], value_mat_S[r,c], value_mat_E[r,c], value_mat_W[r,c] ])
            min_act = np.argmax(mat)
            plt.scatter(c, -r)
            if(min_act ==0):
                rw_ind = imp_dic[(r,c)]
                action_mat[r,c] = min_act
                action_name_mat[r,c] = act_name[min_act]
                new_transition[rw_ind, :] = transition_N[rw_ind, :]
                plt.arrow(c,-r, 0,0.2, width = 0.1)
            if(min_act ==1):
                rw_ind = imp_dic[(r,c)]
                action_mat[r,c] = min_act
                action_name_mat[r,c] = act_name[min_act]
                new_transition[rw_ind, :] = transition_S[rw_ind, :]
                plt.arrow(c,-r, 0,-0.2,  width = 0.1)

            if(min_act ==2):
                rw_ind = imp_dic[(r,c)]
                action_mat[r,c] = min_act
                action_name_mat[r,c] = act_name[min_act]
                new_transition[rw_ind, :] = transition_E[rw_ind, :]
                plt.arrow(c,-r, 0.2,0,  width = 0.1)

            if(min_act ==3):
                rw_ind = imp_dic[(r,c)]
                action_mat[r,c] = min_act
                action_name_mat[r,c] = act_name[min_act]
                new_transition[rw_ind, :] = transition_W[rw_ind, :]
                plt.arrow(c,-r, -0.2,0,  width = 0.1)
    plt.show()


    #init again for policy evaluation
    value_mat = np.reshape(np.linalg.inv(np.identity(100) - gamma*new_transition)@cost_mat.flatten(), (10,10))


# print(new_transition) 
print(action_name_mat)   
 #OVERLAY FIGIURE CREATED HERE
#fig=plt.figure()
#ax=fig.add_subplot(1,1,1)       
img = sns.heatmap(value_mat,annot=True)
plt.show()
