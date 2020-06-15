"""
This file holds all parameters needed to run the model.

Author: Signe Onstad Sævareid
Date: March 28th 2020
File: parameters.py

-----------------------------------------------------------------------------------------------------------------------
* Imported modules: random (generates pseudo-random numbers, https://docs.python.org/3/library/random.html)
                    networkx (networks, version 2.4, https://networkx.github.io/documentation/stable/index.html)
                    numpy (scientific computing, version 1.18.1, https://numpy.org/)
                    math (mathematical functions, https://docs.python.org/3/library/math.html)
-----------------------------------------------------------------------------------------------------------------------
"""

from random import shuffle
import networkx as nx
import numpy as np
import math as m


#######################################################################################################################
# USER INPUT
#######################################################################################################################

##############################
# (1/5) - General
##############################
plot_results = True                 # Boolean, summarizes and plots some of the main results.
log_as_network = True               # Boolean, stores the evolving contacts as a network while running if True.
print_to_console = True             # Boolean, prints simulation updates to the console while running if True.
measure_elapsed_time = True         # Boolean, measures elapsed simulation time if True.

number_of_days = 90                 # Integer, number of simulation days.
number_of_wards = 1                 # Integer, number of wards in the hospital.
beds_per_ward = 19                  # Integer, number of patient beds in a ward.


##############################
# (2/5) - Contacts and shifts
##############################
contact_randomization = 0.0         # Float, share of all contacts randomized withing the ward.
ward_isolation = 1.0               # Float, share of all contacts made within the ward.

shift_assignment = 'percentage'     # String, determines how shifts are assigned, 'random' or 'percentage'.
resting_time = True                 # Boolean, determines whether NUR shift resting time is taken into account.


##############################
# (3/5) - Control measures
##############################
social_distancing = 0.0             # Float, proportion of all contacts being avoided due to social distancing.

t_quarantine = 14                   # Integer, number of quarantine days after last exposure.
t_isolation = 7                     # Integer, number of isolation days after symptoms are gone.
t_milder = 1                        # Integer, number of quarantine days after milder symptoms are gone.

test_results_waiting_time = 36      # Integer, number of hours before test results are available.
close_contact_definition = 20       # Integer, number of contacts in critical window required for notification.
t_notify_contacts = 24              # Integer, number of hours before symptom onset contacts must be notified.
p_remember = 0.5                    # Float, proportion of unique individuals an infected person remembers.

t_quarantine_exception = 14         # Integer, number of quarantine days for HCW if exceptions are allowed.
send_staff_directly_home = True     # Boolean, determines if a Staff entering quarantine finishes its shift of not.


##############################
# (4/5) - COVID-19 infection
##############################
incubation_time = {'mu': 1.6067, 'sigma': 0.4427}       # Dictionary, holding lognormal parameter values for t_E.
asymptomatic_time = {'mu': 1.9459, 'sigma': 0.2980}     # Dictionary, holding lognormal parameter values for t_A.
infection_time = {'mu': 2.6931, 'sigma': 0.4016}        # Dictionary, holding lognormal parameter values for t_I.
infection_ratio = 0.5                                   # Float, relative infection duration of STAFF compared to PAT.

p_asymptomatic = {'STAFF': 0.43, 'PAT': 0.18}       # Dictionary, probabilities of having an asymptomatic infection.
p_death = {'STAFF': 0.00650, 'PAT': 0.327}          # Dictionary, probabilities of fatality given symptomatic infection.

p_vaccination = 0.0         # Float, probability of being vaccinated.
p_efficacy = 0.60           # Float, probability of vaccine providing full immunity.
p_susceptible = 0.01        # Float, probability of not obtaining immunity when recovering from infection.

t_presymptomatic = 2.58     # Float, number of contagious days in incubation period.

alpha = {'N': 0.0, 'A': 0.5, 'I': 1.0}      # Float, degree of infectiousness relative to symptomatic infection.
beta = {'N': 1.0, 'Q': 0.5, 'I': 0.1}       # Float, level of restriction influencing transmission.
gamma = 2                                   # Float, infectiousness compared to influenza virus.

patient_zero = {'n': 2, 'role': 'NUR', 'ward': True}


##############################
# (5/5) Secondary infection
##############################
p_secondary = 0.50                  # Float, probability of acquiring a secondary infection.
p_antibiotics = 0.95                # Float, probability of being treated with antibiotics.
p_resistant = 0.32                  # Float, share of resistant bacterial strains.
p_D_sensitive = 0.19                # Float, case fatality rate in case of susceptible bacterium.
p_D_resistant = 0.68                # Float, case fatality rate in case of susceptible bacterium.


#######################################################################################################################
# CONSTANT VARIABLES
#######################################################################################################################

##############################
# (1/3) Contact and shift variables derived from the empirical data set.
##############################
# Number of contacts per bed per ward for all hours in the day, average and standard deviation.
contact = {0: {'avg': 0.0658, 'std': 0.0996}, 1: {'avg': 0.0395, 'std': 0.0263}, 2: {'avg': 0.2237, 'std': 0.311},
           3: {'avg': 0.25, 'std': 0.2989}, 4: {'avg': 0.0, 'std': 0.0263}, 5: {'avg': 1.2632, 'std': 1.0421},
           6: {'avg': 2.1711, 'std': 0.6203}, 7: {'avg': 9.75, 'std': 0.934}, 8: {'avg': 12.4342, 'std': 0.527},
           9: {'avg': 17.6316, 'std': 1.7931}, 10: {'avg': 23.4342, 'std': 0.9934}, 11: {'avg': 25.1447, 'std': 4.3644},
           12: {'avg': 22.4737, 'std': 2.7607}, 13: {'avg': 11.4000, 'std': 8.8798}, 14: {'avg': 9.9605, 'std': 2.9512},
           15: {'avg': 9.6974, 'std': 5.8741}, 16: {'avg': 9.0263, 'std': 3.8481}, 17: {'avg': 8.0132, 'std': 3.5234},
           18: {'avg': 6.8026, 'std': 3.4388}, 19: {'avg': 3.9605, 'std': 2.8982}, 20: {'avg': 3.9474, 'std': 2.1110},
           21: {'avg': 0.7632, 'std': 0.8728}, 22: {'avg': 1.4211, 'std': 1.328}, 23: {'avg': 0.9342, 'std': 1.101}}

# Parameters for a log-normal distribution giving relative contact rates for individuals from each role.
contact_rates = {'NUR': {'mu': 3.1568, 'sigma': 0.7804, 'upper': 104.2285},
                 'MED': {'mu': 2.6911, 'sigma': 0.6861, 'upper': 42.2250},
                 'ADM': {'mu': 1.6713, 'sigma': 1.4697, 'upper': 27.9692},
                 'PAT': {'mu': 0.7410, 'sigma': 0.6019, 'upper': 6.2969}}


# Proportion of contacts between each role pair.
contact_rate_roles = {'NUR': {'NUR': 0.5000, 'PAT': 0.2782, 'MED': 0.0962, 'ADM': 0.1256},
                      'PAT': {'NUR': 0.7662, 'PAT': 0.0252, 'MED': 0.1497, 'ADM': 0.0589},
                      'MED': {'NUR': 0.2547, 'PAT': 0.1440, 'MED': 0.5332, 'ADM': 0.0681},
                      'ADM': {'NUR': 0.6843, 'PAT': 0.1165, 'MED': 0.1402, 'ADM': 0.0590}}

# Calculated percentages of employment based on five different potential working days in the original data.
percentages_of_employment = {'NUR': [0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.6, 0.6, 0.6,
                                     0.6, 0.6, 0.6, 0.6, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1.0, 1.0, 1.0],
                             'MED': [0.4, 0.4, 0.6, 0.6, 0.8, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0],
                             'ADM': [0.2, 0.4, 0.4, 0.6, 0.6, 0.6, 0.8, 1.0]}


##############################
# (2/3)  Other variables.
##############################
symptoms = {'fever': 0.879, 'cough': 0.677, 'dyspnea': 0.186}   # Dictionary, symptoms and associated probabilities.
hospitalization_duration = {'mean': 147.44, 'std': 52.78}       # Dictionary, variables for duration of hospitalization.
new_shifts = [7, 8, 9, 13, 20]                                  # List, containing hours where new shifts are started.
occupancy = np.divide(16, 19)                                   # Float, bed occupancy in the ward.
min_hospitalization = 8                                         # Integer, minimum hours of PAT hospitalization.
rest = 18                                                       # Integer, hours of rest after a NUR shift.
t0 = 7                                                          # Integer, hour corresponding to first simulation hour.


##############################
# (3/3) Calculated variables
##############################
percentage_of_employment = {'NUR': {}, 'MED': {}, 'ADM': {}}    # Dictionary, holding employment percentages.
initial_regular_staff = {}                                      # Dictionary, holding numbers of staff of each role.
time_steps = []                                                 # List, holding all simulation time steps.
shifts = {}                                                     # Dictionary, holds all information on shifts.


#######################################################################################################################
# DYNAMIC VARIABLES (--> set_variables())
#######################################################################################################################

##############################
# (1/2) Occasional update
##############################
agents_in_simulation = []           # List, for storing all instantiated agents in the simulation.
network = nx.Graph()                # Network, for storing temporal contact information as nodes and links in a network.
tree = nx.DiGraph()                 # Network, for storing infection routes.
available_beds = {}                 # Dictionary, holds number of available beds (value) for each ward (key).
id_number = 0                       # Integer, counts upwards giving all instantiated objects a unique id number.

fatalities = {}                     # Dictionary, updated in case of a death (--> register_new_death()).
individual_carriers = []            # List, updated with new carriers.
individual_cases = []               # List, updated with new confirmed cases.


##############################
# (2/2) Regular update
##############################
carriers = {}                       # Dictionary, holding number of carriers as values for each hour key.
infected_all = {}                   # Dictionary, holding number of infected as values for each hour key.
infected_confirmed = {}             # Dictionary, holding number of confirmed cases as values for each hour key.

cumulative_cases = []               # List, holding cumulative number of cases for each time step.
cumulative_deaths = []              # List, holding cumulative number of deaths for each time step.

agents_incubation = {}              # Dictionary, holding number of agents in incubation phase for each hour key.
agents_asymptomatic = {}            # Dictionary, holding number of agents in asymptomatic infection for each hour key.
agents_symptomatic = {}             # Dictionary, holding number of agents in symptomatic infection for each hour key.

quarantine_agents = {}              # Dictionary, holding number of agents in quarantine for each hour key.
isolation_agents = {}               # Dictionary, holding number of agents in isolation for each hour key.
temporary_staff = {}                # Dictionary, holding number of working temporary staff for each hour key.

number_of_contacts = {}             # Dictionary, holding drawn number of contacts as value for each hour key.


#######################################################################################################################
# METHODS FOR SETTING OR UPDATING PARAMETER VALUES
#######################################################################################################################
def set_variables():
    """Sets variables depending on other input variables, based on values derived from the empirical data set."""

    ##############################
    # (1/2) Constant variables
    ##############################

    global percentage_of_employment, initial_regular_staff, time_steps, shifts

    if shift_assignment == 'percentage':
        for role, percentages in percentages_of_employment.items():
            for percentage in set(percentages):
                percentage_of_employment[role][percentage] = np.divide(percentages.count(percentage), len(percentages))
    else:
        for role in percentages_of_employment.keys():
            percentage_of_employment[role] = {1.0: 1.0}

    initial_regular_staff = {'NUR': int(27 / 19 * beds_per_ward),
                             'MED': int(11 / 19 * beds_per_ward),
                             'ADM': int(8 / 19 * beds_per_ward)}

    time_steps = list(np.arange(t0, t0 + number_of_days * 24))

    shifts = {'NUR': {7: {'dur': 7, 'num': int(9 / 19 * beds_per_ward), 'type': 'morning'},
                      9: {'dur': 8, 'num': int(2 / 19 * beds_per_ward), 'type': 'day'},
                      13: {'dur': 7, 'num': int(5 / 19 * beds_per_ward), 'type': 'afternoon'},
                      20: {'dur': 11, 'num': int(2 / 19 * beds_per_ward), 'type': 'night'}},
              'MED': {9: {'dur': 10, 'num': int(8 / 19 * beds_per_ward)}},
              'ADM': {8: {'dur': 10, 'num': int(4 / 19 * beds_per_ward)}}}

    ##############################
    # (1/2) Dynamic variables
    ##############################

    global agents_in_simulation, network, tree, available_beds, id_number
    agents_in_simulation = []
    network = nx.Graph()
    tree = nx.DiGraph()
    available_beds = {}
    id_number = 0

    for ward in range(number_of_wards):
        update_available_beds(ward + 1, int(round(np.multiply(beds_per_ward, occupancy))))

    global fatalities, individual_carriers, individual_cases
    fatalities = {}
    individual_carriers = []
    individual_cases = []

    global carriers, infected_all, infected_confirmed
    carriers = {i: 0 for i in time_steps}
    infected_all = {i: 0 for i in time_steps}
    infected_confirmed = {i: 0 for i in time_steps}

    global cumulative_cases, cumulative_deaths
    cumulative_cases, cumulative_deaths = [0] * len(time_steps), [0] * len(time_steps)

    global agents_incubation, agents_asymptomatic, agents_symptomatic
    agents_incubation = {i: 0 for i in time_steps}
    agents_asymptomatic = {i: 0 for i in time_steps}
    agents_symptomatic = {i: 0 for i in time_steps}

    global quarantine_agents, isolation_agents, temporary_staff
    quarantine_agents = {i: 0 for i in time_steps}
    isolation_agents = {i: 0 for i in time_steps}
    temporary_staff = {i: 0 for i in time_steps}

    global number_of_contacts
    number_of_contacts = {}


def summarize_time_step(hour):
    """
    Summarizes each time step by counting agents in different epidemiological states.
    :param hour: Integer, marking the time step.
    :return: None
    """

    agents_in_infection = [a for a in agents_in_simulation if a.infection > 0]
    staff = [a for a in agents_in_simulation if a.role == 'NUR' or a.role == 'MED' or a.role == 'ADM']

    # Carriers, infected and confirmed infected
    carriers[hour] = sum(map(lambda x: x.carrier, agents_in_simulation))
    infected_all[hour] = len(agents_in_infection)
    infected_confirmed[hour] = sum(map(lambda x: x.positive_test > 0, agents_in_infection))

    # Cumulative numbers
    cumulative_deaths[hour:] = [sum(map(lambda x: not x.alive, agents_in_simulation))] * len(cumulative_deaths[hour:])
    cumulative_cases[hour:] = [sum(map(lambda x: x.positive_test, agents_in_simulation))] * len(cumulative_cases[hour:])

    # Incubation, asymptomatic and symptomatic infections
    agents_incubation[hour] = sum(map(lambda x: x.incubation > 0, agents_in_simulation))
    agents_asymptomatic[hour] = sum(map(lambda x: x.infectious == 'A', agents_in_infection))
    agents_symptomatic[hour] = sum(map(lambda x: x.infectious == 'I', agents_in_infection))

    # Quarantine, isolation and temporary staff
    temporary_staff[hour] = sum(map(lambda x: (x.currently_present and not x.regular_staff), staff))
    quarantine_agents[hour] = sum(map(lambda x: x.restriction['level'] == 'Q', agents_in_simulation))
    isolation_agents[hour] = sum(map(lambda x: x.restriction['level'] == 'I', agents_in_simulation))


def add_new_agent(agent):
    """
    Adds a new agent to the simulation, storing it in the list agents_in_simulation.

    :param agent: Any object inheriting from Individual.
    :return: None
    """

    global agents_in_simulation
    agents_in_simulation.append(agent)

    if log_as_network:
        add_node_to_networks(agent.role, agent.ward)


def update_available_beds(ward, change):
    """
    Updates the number of available beds in a ward.

    :param ward: Integer, number indicating to which ward the bed.
    :param change: Integer, either a negative positive number indicating the change in number of available beds.
    :return: None
    """

    global available_beds

    if ward not in available_beds:
        available_beds[ward] = change
    else:
        available_beds[ward] += change
        if available_beds[ward] < 0:
            print("Cannot have negative number of available beds!")


def update_id_number():
    """Updates the id_number by counting upwards."""

    global id_number
    id_number += 1


def get_log_normal_parameters(mode, min_value, max_value, sum_of_integral):
    """
    Calculates parameters mu and sigma for a log-normal distribution based on desired mode.

    :param mode: The value to appear most often, represents the top of the curve.
    :param min_value: Minimum value in the distribution, set to 1E-10 if set to 0.
    :param max_value: Maximum value in the distribution.
    :param sum_of_integral: Sum of the area under the distribution curve between min_value and max_value.
    :return: parameters: Dictionary containing the mu and sigma values representing the incubation period distribution.
    """

    min_value = min_value if min_value > 0 else 1E-10

    if mode <= min_value or mode >= max_value:
        raise ValueError('The mode {} does not lie within the range ({}, {}) days'.format(mode, min_value, max_value))

    t = {}
    sigma_values = np.arange(0.0001, 1, 0.0001)
    for s in sigma_values:
        mu = np.log(mode) + s ** 2

        cdf = (0.5 + 0.5 * m.erf(np.divide((np.log(max_value) - mu), (2 ** 0.5 * s)))) \
            - (0.5 + 0.5 * m.erf(np.divide((np.log(min_value) - mu), (2 ** 0.5 * s)))) \
            - sum_of_integral

        t[s] = abs(cdf)

    sigma = min(t, key=t.get)
    mu = round(np.log(mode) + sigma ** 2, 4)
    parameters = {'mu': mu, 'sigma': sigma}

    return parameters


def add_node_to_networks(role, ward):
    """
    Adds a node to the network having role and ward as attributes.

    :param role: String, three letter code denoting role in the ward ('NUR', 'MED', 'ADM', 'PAT').
    :param ward: Integer, number indicating to which ward the Individual belongs.
    :return: None
    """

    network.add_node(id_number, role=str(role), ward=ward)


def add_edge_to_network(agent, neighbor, contacts, h):
    """
    Adds or updates an edge between two Individual-inheriting objects in the network.

    :param agent: Individual-inheriting objects, person A.
    :param neighbor: Individual-inheriting objects, person B.
    :param contacts: Integer, number of contacts in the given hour.
    :param h: Integer, simulation hour + h.
    :return: None
    """

    number, listed = contacts, [h] * contacts

    # If the edge does not exist, it is added.
    if not network.has_edge(agent.id_number, neighbor.id_number):
        network.add_edge(agent.id_number, neighbor.id_number, number_contacts=number, listed_contacts=listed)

    # If the edge does exist, it is updated.
    else:
        existing_attributes = network.get_edge_data(agent.id_number, neighbor.id_number)
        s_contacts = existing_attributes['number_contacts'] + number
        s_listed = existing_attributes['listed_contacts'] + listed
        network.add_edge(agent.id_number, neighbor.id_number, number_contacts=s_contacts, listed_contacts=s_listed)


def add_infection_route_to_tree(agent, neighbor, hour):
    """
    Adds a branch to the tree of infection routes.

    :param agent: Individual-inheriting objects, infector.
    :param neighbor:  Individual-inheriting objects, infectee.
    :param hour: Integer, time point of which the transmission occurs.
    :return: None
    """

    tree.add_node(neighbor.id_number, role=str(neighbor.role), ward=neighbor.ward,
                  extroversion=neighbor.extroversion, hour=hour)

    tree.add_edges_from([(agent.id_number, neighbor.id_number)], hour=str(hour))


def register_new_death(agent_id, hour, role, cause):
    """
    Registers a new COVID-19 death.
    :param agent_id: Integer, a unique id number identifying the agent.
    :param role: String, three letter code denoting role in the ward ('NUR', 'MED', 'ADM', 'PAT').
    :param hour: Integer, time point of death.
    :param cause: String, primary or secondary infection.
    :return: None
    """

    fatalities[agent_id] = {'hour': hour, 'role': role, 'cause': cause}


def shuffle_list_of_agents():
    """Shuffles the list of agents in the simulation."""

    shuffle(agents_in_simulation)


########################################################################################################################
# METHODS FOR PRINTING
########################################################################################################################

def print_message(message, elapsed_time=None):
    """
    Prints a message to console, with or without the elapsed time.

    :param message: String, message to be printed to console.
    :param elapsed_time: Float, seconds of elapsed running time.
    :return: None
    """

    if elapsed_time is not None:
        print("-------------------------------------------------------------------------------------------------------")
        print("{} Elapsed time: {} s".format(message, int(round(elapsed_time, 2))))
        print("-------------------------------------------------------------------------------------------------------")

    else:
        print("-------------------------------------------------------------------------------------------------------")
        print("{}".format(message))
        print("-------------------------------------------------------------------------------------------------------")


def progress(iteration, total, bar_length=80):
    """
    Generates a terminal progress bar when called in a loop.
    --> Adapted from https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a

    :param iteration: Integer, the current iteration.
    :param total: Integer, the number of total iterations.
    :param bar_length: Integer, character length of bar.
    :return: None
    """

    percents = f'{int(round(100 * (iteration / float(total)))):}'
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = f'{"█" * filled_length}{"-" * (bar_length - filled_length)}'
    print(f'\r\t- Progress: |{bar}| {percents}%', end=''),
    if iteration == total:
        print("")


########################################################################################################################