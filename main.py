# here put the import lib
import os
import pdb
import time
import random 
import shutil
import numpy as np

import openmc
from neorl import JAYA
from neorl import PPO2
from neorl import MlpPolicy
from neorl import RLLogger
from neorl import CreateEnvironment
from neorl import PPOES
from pathlib import Path


opt_algorithm = 'JAYA'  # JAYA/PPO-ES
ncores = 1 # parallel computing number
verbose = True
ngen = 100
seed= 100

start = time.time()
cur_path = os.getcwd()
print('Current working path:', cur_path)
logger_path = os.path.join(f'{cur_path}/logs', f'my_history_{opt_algorithm}.txt')

## Configure enviromental variable here ##
os.environ['OPENMC_CROSS_SECTIONS'] = '/tmp/endfb-vii.1-hdf5/cross_sections.xml'

my_limit = 0.1685 # north south limit
mesh_parameter_x = 20
mesh_parameter_y = 7
U_density, water_2_density = 19, 1
lb = [0.1, 0.001]  # [0.3, 0.1]
ub = [19, 30]  # [19.0, 20]
d2type = ['float', 'float']
nx = 2

def create_bounds(lb, ub, d2type, nx):
    bounds = {}
    for i in range(nx):
        bounds['x' + str(i + 1)] = [d2type[i], lb[i], ub[i]]
    return bounds

def remove_directory(path):
    os.system(f'rm -r "{path}" && rmdir "{path}"')

# Check if the file exists
if os.path.exists(logger_path):
    # If the file exists, delete it
    os.remove(logger_path)
    print(f"The file {logger_path} has been deleted.")
else:
    print(f"The file {logger_path} does not exist.")

def writing_function(content):
    with open(logger_path, 'a+') as file:
        file.write(f'-{str(content)}\n')

def init_model(U_density, water_2_density):
    model = openmc.model.Model()
    # Define materials
    o16 = openmc.Nuclide('O16')
    h1 = openmc.Nuclide('H1')

    # Instantiate Materials
    aluminum_a5 = openmc.Material(material_id=20, name='Aluminum A5 (clad)')
    aluminum_a5.add_nuclide('Al27', 5.8339E-02)
    aluminum_a5.add_element('Fe', 7.2563E-05)
    aluminum_a5.add_element('Si', 5.8050E-05)
    aluminum_a5.add_element('Cu', 2.0477E-06)
    aluminum_a5.add_element('Mg', 1.0033E-05)
    aluminum_a5.add_nuclide('B10', 3.2508E-07)
    aluminum_a5.add_nuclide('B11', 1.1821E-06)
    aluminum_a5.add_nuclide('Li6', 1.9640E-07)
    aluminum_a5.add_nuclide('Li7', 2.1537E-06)
    aluminum_a5.add_element('Cd', 1.4384E-07)

    meat101 = openmc.Material(material_id=21, name='UAL meat 93% HEU 10ppm Boron')
    meat101.add_nuclide('U235', 1.658E-03)
    meat101.add_nuclide('U238', 1.248E-04)
    meat101.set_density('g/cm3', U_density)

    water = openmc.Material(material_id=22, name='Water at 300 K')
    water.set_density('g/cm3', 0.1)
    water.add_nuclide('O16', 1.0)
    water.add_nuclide('H1', 2.0)
    water.add_s_alpha_beta('c_H_in_H2O') #I added this 18.12.23

    aluminum_ag3= openmc.Material(material_id=23, name='name_Aluminum_AG3')
    aluminum_ag3.add_nuclide('Al27',5.8273E-02)
    aluminum_ag3.add_element('Fe',  7.2563E-05)
    aluminum_ag3.add_element('Si',  8.7075E-05)
    aluminum_ag3.add_element('Cu',  2.0477E-06)
    aluminum_ag3.add_element('Mg',  1.8060E-03)
    aluminum_ag3.add_element('Cr',  9.3773E-06)
    aluminum_ag3.add_nuclide('Mn55',8.8658E-06)
    aluminum_ag3.add_nuclide('B10', 3.2508E-07)
    aluminum_ag3.add_nuclide('B11', 1.1821E-06)
    aluminum_ag3.add_nuclide('Li6', 1.9640E-07)
    aluminum_ag3.add_nuclide('Li7', 2.1537E-06)
    aluminum_ag3.add_element('Cd',  1.4384E-07)

    water_2 = openmc.Material(24, "h2o")
    water_2.add_nuclide(h1, 2.0)
    water_2.add_nuclide(o16, 1.0)
    water_2.set_density('g/cm3', water_2_density)
    water_2.remove_nuclide('O16')
    water_2.add_element('O', 1.0)
    water_2.add_s_alpha_beta('c_H_in_H2O')

    Cadnium = openmc.Material(25, "cd")
    Cadnium.add_element('Cd', 1.0)

    # Define the materials file.
    model.materials = (meat101, aluminum_ag3, aluminum_a5, water, Cadnium, water_2)

    # Instantiate Surfaces
    meat_north = openmc.YPlane(y0=0.0255, name='meat north side')
    meat_south = openmc.YPlane(y0=-0.0255, name='meat south side')
    meat_east  = openmc.XPlane(x0=3.115, name='meat south side')
    meat_west  = openmc.XPlane(x0=-3.115, name='meat south side')

    clad_north = openmc.YPlane(y0=0.0635, name='clad north side')
    clad_south = openmc.YPlane(y0=-0.0635, name='clad south side')
    clad_east  = openmc.XPlane(x0=3.33, name='clad south side')
    clad_west  = openmc.XPlane(x0=-3.33, name='clad south side')
    #
    east_side_plate_east  = openmc.XPlane(x0=3.805, name='side plate east side')
    west_side_plate_west  = openmc.XPlane(x0=-3.805, name='side plate west side')
    #
    #
    east   = openmc.XPlane(x0= 4.2, name='boundery right') #x0= 4.355
    west   = openmc.XPlane(x0=-4.2,name='boundery left') #x0=-3.855
    north  = openmc.YPlane(y0= my_limit, name='boundery front')
    south  = openmc.YPlane(y0= -my_limit, name='boundery back')
    east.boundary_type   = 'reflective'
    west.boundary_type   = 'reflective'
    north.boundary_type  = 'reflective'
    south.boundary_type  = 'reflective'

    # Instantiate Cells
    meat        = openmc.Cell(cell_id=701, name='cell 701')
    side_plates = openmc.Cell(cell_id=703, name='cell 703')
    coolant     = openmc.Cell(cell_id=704, name='cell 704')
    right_trap = openmc.Cell(cell_id=705, name='cell 705')
    left_trap = openmc.Cell(cell_id=706, name='cell 706')
    clad        = openmc.Cell(cell_id=702, name='cell 702')
    #
    # Use surface half-spaces to define regions
    meat.region        = +meat_south & -meat_north & +meat_west & -meat_east
    clad.region        = +clad_south & -clad_north & +clad_west & -clad_east & ~meat.region
    side_plates.region = (( +clad_east & -east_side_plate_east ) | ( +west_side_plate_west  & -clad_west )) & -north & +south
    coolant.region     = +clad_west & -clad_east & (+clad_north | -clad_south) & -north & +south
    right_trap.region = (+west & -west_side_plate_west) & -north & +south
    left_trap.region = (+east_side_plate_east & -east) & -north & +south

    #uo2, zirconium, water,
    # Register Materials with Cells
    coolant.fill     = water_2
    right_trap.fill = water_2
    left_trap.fill = Cadnium
    meat.fill        = meat101
    side_plates.fill = aluminum_ag3
    clad.fill        = aluminum_a5

    # Instantiate root universe
    model.geometry.root_universe = openmc.Universe(universe_id=10, name='root universe')

    # Register cells with Universe
    model.geometry.root_universe.add_cells([meat,clad, side_plates, left_trap, right_trap, coolant])
    model.settings.batches =  100  # OpenMC total calculating batches
    model.settings.inactive = 30  # the first 30 batches are inactive
    model.settings.particles = 1000  # number of simulation particles
    model.settings.source = openmc.Source(space=openmc.stats.Box(
        [-1 / 2, -1 / 2, -1], [1 / 2, 1 / 2, 1], only_fissionable=True))  # define the source
    # Instantiate an empty Tallies object
    #tallies_file = openmc.Tallies()
    model.tallies = openmc.Tallies()
    mesh = openmc.RegularMesh()
    mesh.dimension = [mesh_parameter_x, mesh_parameter_y]
    mesh.lower_left = [- 4.2, - 0.1685]
    mesh.upper_right = [4.2, 0.1685]
    mesh_filter = openmc.MeshFilter(mesh)
    tally = openmc.Tally(name='flux')
    energy_filter = openmc.EnergyFilter([0.0, 0.6, 1.0e6])
    tally.filters = [mesh_filter, energy_filter]
    tally.scores = ['flux', 'fission']
    model.tallies.append(tally)
    return model, meat101, water_2


def update_one_plate(U_density, water_2_density, meat101, water_2):
    meat101.set_density('g/cm3', U_density)
    water_2.set_density('g/cm3', water_2_density)

    # Update Materials right_trap
    model.materials[0] =meat101
    model.materials[-1] = water_2
    ##################
    # Update Geometry
    model.geometry.root_universe.cells[701].fill = meat101  # Update meat fill with new U density
    model.geometry.root_universe.cells[704].fill = water_2  # Update trap fill with water 2 density
    model.geometry.root_universe.cells[705].fill = water_2
    ##################

# call NEORL to find the optimal enrichment ##
# Define the fitness function
def FIT(x):
    print("x[0]: ", '%12.0f' % x[0])
    print("x[1]: ", '%12.0f' % x[1])
    # create a subfold for parallel computing
    randnum = random.randint(0,1e8) # create a random number
    pathname = os.path.join(cur_path, 'PPoES_subfold_11_'+str(randnum)) # create subfold
    os.makedirs(pathname, exist_ok=True)
    os.chdir(pathname) # change working dir into the subfold

    # OpenMC calculation
    update_one_plate(U_density=x[0], water_2_density=x[1], meat101=meat101, water_2=water_2)
    result_r = model.run(output=True) # path of h5 file
    sp = openmc.StatePoint(result_r) # State information on a simulation
   
    # From here is my part
    sp2 = openmc.StatePoint('statepoint.100.h5')
    tally = sp2.get_tally(scores=['flux'])
    flux = tally.get_slice(scores=['flux'])
    fission = tally.get_slice(scores=['fission'])
    print(flux)
    print(fission)
    flux.std_dev.shape = (mesh_parameter_y, mesh_parameter_x, 2)
    flux.mean.shape = (mesh_parameter_y, mesh_parameter_x, 2)
    fission.std_dev.shape = (mesh_parameter_y, mesh_parameter_x, 2)
    fission.mean.shape = (mesh_parameter_y, mesh_parameter_x, 2)
    my_thermal_flux_sum = []
    my_fast_flux_sum = []
    for i in range(mesh_parameter_x):
        my_thermal_flux_sum.append(np.average(flux.mean[0:mesh_parameter_y, i, 0]))
        my_fast_flux_sum.append(np.average(flux.mean[0:mesh_parameter_y, i, 1]))
    k_combined = sp.k_combined # the combined k-eff
    k_combined_nom = k_combined.nominal_value # the nominal value of k-eff
    k_combined_stddev = k_combined.std_dev # the standard deviation of k-eff
    return_val =  (1+10*abs(k_combined_nom - 1))/(flux.mean[0, 0, 0]+0.000001)
    writing_function([x[0], x[1], my_thermal_flux_sum[0],  my_fast_flux_sum[0], k_combined_nom, return_val])  #TODO: Write to txt file only in the end
    return return_val


model, meat101, water_2 = init_model(U_density=U_density, water_2_density=water_2_density)
if opt_algorithm =='JAYA':
    BOUNDS = create_bounds(lb, ub, d2type, nx)
    #--setup the parameter space
    # use JAYA to find the optimal U enrichment
    jaya=JAYA(mode='min', bounds=BOUNDS, fit=FIT, npop=4, ncores=ncores, seed=seed)
    x_best, y_best, jaya_hist=jaya.evolute(ngen=ngen, verbose=verbose)
    print('---JAYA Results---', )
    print('x:', x_best)
    print('y:', y_best)
    print('JAYA History:\n', jaya_hist)

    end = time.time()
    running_time = end - start
    print('running time:\n', running_time)

if opt_algorithm=='PPO-ES':
    BOUNDS = create_bounds(lb, ub, d2type, nx)
    # create an enviroment class
    env=CreateEnvironment(method='ppo',
                          fit=FIT,ncores=ncores,
                          bounds=BOUNDS,
                          mode='min',
                          episode_length=50)

    #change hyperparameters of PPO/ES if you like (defaults should be good to start with)
    h={'cxpb': 0.8,'mutpb': 0.2,'n_steps': 24,'lam': 1.0}

    #Important: `mode` in CreateEnvironment and `mode` in PPOES must be consistent
    #fit is needed to be passed again for ES, must be same as the one used in env
    ppoes=PPOES(mode='min',
                fit=FIT,
                env=env,
                npop_rl=4,
                init_pop_rl=True,
                bounds=BOUNDS,
                hyperparam=h,
                seed=seed)
    #first run RL for some timesteps
    rl=ppoes.learn(total_timesteps=3000, verbose=verbose)
    #second run ES, which will use RL data for guidance
    ppoes_x, ppoes_y, ppoes_hist=ppoes.evolute(ngen=ngen, ncores=ncores, verbose=verbose)
    print('---PPO-ES Results---', )
    print('x:', ppoes_x)
    print('y:', ppoes_y)
    print('PPO-ES History:\n', ppoes_hist)