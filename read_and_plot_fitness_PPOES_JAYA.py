import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

plt.figure()
for algorithm_num in range(2):
    if algorithm_num == 0:
        #file_path = "/home/deborahsw/Desktop/projects/coreai/my_history_mmg_plate_with_CM_JAYA_7b_2_24_only_k.txt"
        file_path = "/home/deborahsw/Desktop/projects/RLOpenNeoMC/logs/my_history_PPO-ES.txt"
    if algorithm_num == 1:
        # file_path = "/home/deborahsw/Desktop/projects/coreai/my_history_mmg_plate_with_CM_JAYA_with_18_3_b_model_new_f.txt"
        # file_path = "/home/deborahsw/Desktop/projects/coreai/my_history_mmg_plate_with_CM_JAYA_7b_2_24_only_k.txt"
        file_path = "/home/deborahsw/Desktop/projects/RLOpenNeoMC/logs/my_history_JAYA.txt"
        # file_path = "/home/deborahsw/Desktop/projects/coreai/my_history_mmg_plate_with_CM_PPOES_5_2_24_only_k.txt"
        # Initialize vectors
    U_dens = []
    W_dens = []
    thermal_flux = []
    fast_flux = []
    k = []
    grad = []

    # Read the file
    with open(file_path, 'r') as file:
        for line in file:
            # Remove leading and trailing whitespace
            line = line.strip()

            # Check if the line starts with '-[' and ends with ']'
            if line.startswith('-[') and line.endswith(']'):
                # Extract the content between '-[' and ']'
                content = line[2:-1]

                # Split the content into a list of strings
                values = content.split(', ')

                # Convert the strings to float numbers
                values = [float(val) for val in values]

                # Assign values to the appropriate vector
                U_dens.append(values[0])
                W_dens.append(values[1])
                thermal_flux.append(values[2])
                fast_flux.append(values[3])
                k.append(values[4])
                grad.append(values[5])
                temporaly_min = []
                time = []
                min_vec = []
                min = 100
                for i, value in enumerate(grad):
                    if value < min:
                        min = value
                        min_vec.append(min)
                        time.append(i)
                        minimal_index=i

                min_vec.append(min)
                time.append(len(grad))

                my_x = []
                my_y = []
                for i in range(len(min_vec)-1):
                    my_y.append(min_vec[i])
                    my_x.append(time[i])

                    my_y.append(min_vec[i])
                    my_x.append(time[i+1])

                # my_y.append(min_vec[len(min_vec)-1])
                # my_x.append(time[len(min_vec)-1])


        #print('the maximal value = ', temporaly_max[i])
    if algorithm_num == 0:
        plt.plot(my_x, my_y,  '.', c='b', label = "PPOES")
        plt.plot(my_x, my_y, c='b')
        print('time PPOES =', len(grad))
    if algorithm_num == 1:
        plt.plot(my_x, my_y, '.', c='r', label="JAYA")
        plt.plot(my_x, my_y, c='r')
        print('time JAYA =', len(grad))
    print('U_dens=', U_dens[minimal_index])
    print('W_dens=', W_dens[minimal_index])
    print('fast_flux=', fast_flux[minimal_index])
    print('thermal_flux=', thermal_flux[minimal_index])
    print('k=', k[minimal_index])
    print('grad=', grad[minimal_index])

    my_calculation1 = (1+10*abs(k[minimal_index]-1))/(fast_flux[minimal_index]+0.000001)
    my_calculation2 = (1 + 10 * abs(k[minimal_index] - 1)) / (thermal_flux[minimal_index] + 0.000001)
    print('grad_fast=', my_calculation1)
    print('grad_thermal=', my_calculation2)
    print ('\n\n')

# plt.xlim((0, 250))
# plt.ylim((0, 10))
plt.xlabel('Step', fontsize=20) # , fontsize=18
plt.ylabel('fitness',  fontsize=20)  #('Best Tour Cost')
plt.legend(fontsize=12,  loc='upper right')
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
# plt.text(300, 9,'(a)', fontsize = 18)
plt.show()
plt.savefig("/home/deborahsw/Desktop/projects/coreai/JAYA_PPOES_17_3.png", format='png', dpi=300, bbox_inches="tight")