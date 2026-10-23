import random

size = 10
intial_field = [[0 for _ in range(size)] for _ in range(size)]

# creating life 
intial_field[3][4] = 1 
intial_field[2][4] = 1 
intial_field[1][4] = 1 

intial_field[3][8] = 1 
intial_field[2][6] = 1 
intial_field[1][5] = 1 
intial_field[8][8] = 1 
intial_field[8][6] = 1 
intial_field[9][5] = 1 

# displaying the field 
for i in range(size):
    for j in range(size):
        print(intial_field[i][j] , end=" ")
    print("")


def nex_phase(field):
    #    take the field and duplicate it
    new_field = field
    # check whether the neighbours of the alive cells 
    
    for row in new_field:
        for col in row:
            # check if it's a edge or a corner 
            
    # Mark them as alive/dead
        # 
    # Make a new grid - for every cell 
        # check if they should survive or not
        # Mark one/zero 
    # mext_fie
    pass