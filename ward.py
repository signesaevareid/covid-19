"""
This file runs one or several simulations of the spread of SARS-CoV-2 in one or more virtual hospital wards.  by setting variables, initializing agents and generating a temporal network facilitating
inter-individual spread of SARS-CoV-2.

Author: Signe Onstad Sævareid
Date: March 6th 2020
File: ward.py

Written for Python 3.7.3
-----------------------------------------------------------------------------------------------------------------------
* Imported modules: matplotlib.ticker (tick formation, version 3.2.1, https://matplotlib.org/3.2.1/api/ticker_api.html)
                    matplotlib.pyplot (visualization, version 3.1.2, https://matplotlib.org/3.2.1/)
                    scipy.stats (statistics, version 1.4.1, https://docs.scipy.org/doc/scipy-1.4.1/reference/)
                    networkx (networks, version 2.4, https://networkx.github.io/documentation/stable/index.html)
                    numpy (scientific computing, version 1.18.1, https://numpy.org/)
                    os.path (pathnames, https://docs.python.org/3/library/os.path.html)
                    time (time-related functions, https://docs.python.org/3/library/time.html)

* Imported files: parameters.py
                  classes.py
-----------------------------------------------------------------------------------------------------------------------
"""


from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import scipy.stats as stats
import matplotlib as mpl
import networkx as nx
import numpy as np
import os.path
import time

import parameters
import classes


def options():
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    plt.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    plt.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'


class_names = {'MED': classes.MED, 'ADM': classes.ADM, 'NUR': classes.NUR, 'PAT': classes.PAT}


#######################################################################################################################
# SET VARIABLES
#######################################################################################################################

def set_variables():
    """Sets all model variables, including class variables."""

    parameters.set_variables()
    set_class_variables()

    if parameters.print_to_console:
        print(u"\u2713\tAll variables are set.")


############################################################

def set_class_variables():
    """Sets class variables from parameter input."""

    # Susceptible (S)
    classes.Individual.p_V = parameters.p_vaccination
    classes.Individual.p_E = parameters.p_efficacy

    # Exposed (E)
    classes.Individual.incubation_period = parameters.incubation_time
    classes.Individual.t_presymptomatic = int(round(24 * parameters.t_presymptomatic))
    classes.Staff.p_A = parameters.p_asymptomatic['STAFF']
    classes.PAT.p_A = parameters.p_asymptomatic['PAT']

    # Asymptomatic (A)
    classes.Individual.t_A = parameters.asymptomatic_time

    # Infection (I)
    mu_pat, sigma_pat = parameters.infection_time['mu'], parameters.infection_time['sigma']
    mu_staff = np.log(parameters.infection_ratio * np.exp(mu_pat))
    sigma_staff = np.sqrt(mu_staff - np.log(parameters.infection_ratio * np.exp(mu_pat - sigma_pat ** 2)))

    classes.Individual.symptoms = parameters.symptoms
    classes.Staff.t_I = {'mu': mu_staff, 'sigma': sigma_staff}
    classes.PAT.t_I = parameters.infection_time
    classes.Staff.p_D = parameters.p_death['STAFF']
    classes.PAT.p_D = parameters.p_death['PAT']

    # Recovered (R)
    classes.Individual.p_susceptible = parameters.p_susceptible

    # Control measures
    classes.Individual.t_quarantine = parameters.t_quarantine
    classes.HealthcareWorker.t_quarantine = parameters.t_quarantine_exception
    classes.Individual.t_isolation = parameters.t_isolation

    # Bacterial
    classes.PAT.p_secondary = parameters.p_secondary
    classes.PAT.p_antibiotics = parameters.p_antibiotics
    classes.PAT.p_resistant = parameters.p_resistant
    classes.PAT.p_D_sensitive = parameters.p_D_sensitive
    classes.PAT.p_D_resistant = parameters.p_D_resistant


#######################################################################################################################
# AGENT INITIALIZATION
#######################################################################################################################

def initialize_agents():
    """Initializes a set of staff and patients per ward, stored in the list agents_in_simulation."""

    # Staff
    for ward in range(parameters.number_of_wards):
        for role, total_number in parameters.initial_regular_staff.items():
            for i in range(total_number):
                hire_staff(ward=ward + 1, role=role, hour=0)

    # Patients
    for i in range(int(round(parameters.beds_per_ward * parameters.number_of_wards * parameters.occupancy))):
        admit_patient(hour=0)

    # Initialize patient zero
    initially_infected = set_patient_zero()

    if parameters.print_to_console:
        print(u"\u2713\tAll agents are initialized.")
        print("\t- Initial carriers of SARS-CoV-2:", end=' ')
        print(*initially_infected, sep=', ')


############################################################

def hire_staff(ward, role, hour, regular_staff=True):
    """
    Instantiates a new Staff-inheriting object with a given role.

    :param ward: Integer, ward number.
    :param role: String, one of the given staff roles 'NUR', 'MED' or 'ADM'.
    :param hour: Integer, day hour marking the time point where new staff is hired.
    :param regular_staff: Boolean, marks the difference between the regular staff and temporary substitutes.
    :return: hired: Object inheriting from the Staff class.
    """

    parameters.update_id_number()

    hired = class_names[role](id_number=parameters.id_number,
                              ward=ward,
                              extroversion=get_relative_contact_rate(role),
                              percentage=get_percentage(role),
                              regular_staff=regular_staff)

    parameters.add_new_agent(hired)

    return hired


def admit_patient(hour):
    """
    Admits a patient to the ward in the given hour.

    :param hour: Integer, simulation hour + t0.
    :return: None
    """

    parameters.update_id_number()
    ward = max(parameters.available_beds, key=parameters.available_beds.get)
    parameters.update_available_beds(ward, -1)

    patient = classes.PAT(id_number=parameters.id_number,
                          ward=ward,
                          extroversion=get_relative_contact_rate('PAT'),
                          admission=hour,
                          hours_left_in_ward=get_hours_of_hospitalization(hour))

    parameters.add_new_agent(patient)


def set_patient_zero():
    """
    Randomly infects a number of susceptible individuals belonging to a given role.

    :param: None.
    :return: initially_infected: List, containing the Individual-inheriting objects being initial carriers of the virus.
    """

    # Finds all susceptible individuals fulfilling the role requirement.
    number, role = parameters.patient_zero['n'], parameters.patient_zero['role']
    potential_candidates = [a for a in parameters.agents_in_simulation if a.role == role if a.susceptible]

    # If the patient zero individuals are to come from the same ward, this ward is set to ward 1.
    if parameters.patient_zero['ward']:
        potential_candidates = [a for a in potential_candidates if a.ward == 1]

    # The number is adjusted if not enough individuals are available.
    n = number if len(potential_candidates) >= number else len(potential_candidates)
    initially_infected = np.random.choice(potential_candidates, n, replace=False)
    for individual in initially_infected:
        individual.catch_virus(parameters.t0)
        individual.patient_zero = True

        if parameters.log_as_network:
            parameters.tree.add_node(individual.id_number, role=str(individual.role), ward=individual.ward,
                                     extroversion=individual.extroversion, hour=0)

    return initially_infected


##############################

def get_relative_contact_rate(role):
    """
    Draws relative contact rate from a log-normal distribution depending on the given role.

    :param role: String, category in the hospital ward, 'MED', 'ADM', 'NUR' or 'PAT'.
    :return: extroversion: Float, number of contacts per hour per potential neighbors.
    """

    valid = False

    while not valid:
        mu, sigma = parameters.contact_rates[role]['mu'], parameters.contact_rates[role]['sigma']
        extroversion = np.random.lognormal(mu, sigma, size=None)
        if extroversion <= parameters.contact_rates[role]['upper']:
            valid = True

    return extroversion


def get_percentage(role):
    """
    Draws a percentage of employment for the given role, or 1.0 if random shift assignment.

    :param role: String, one of the given staff roles, 'MED', 'ADM' or 'NUR'.
    :return: percentage: Float, percentage of employment.
    """

    percentage = np.random.choice(list(parameters.percentage_of_employment[role].keys()), None,
                                  p=list(parameters.percentage_of_employment[role].values())) \
        if parameters.shift_assignment == 'percentage' else 1.0

    return percentage


def get_hours_of_hospitalization(hour):
    """
    Draws duration of hospitalization from a truncated normal distribution.

    :param hour: Integer, simulation hour + t0.
    :return: hours_left: Integer, remaining hour of hospitalization.
    """

    # Number of hospitalization hours is drawn form a truncated normal distribution.
    mean, std = parameters.hospitalization_duration['mean'], parameters.hospitalization_duration['std']
    # a, b = np.divide(parameters.min_hospitalization - mean, std), np.divide(np.inf - mean, std)
    length_of_hospitalization = int(round(stats.norm.rvs(loc=mean, scale=std, size=None)))
    length_of_hospitalization = length_of_hospitalization \
        if length_of_hospitalization >= parameters.min_hospitalization else parameters.min_hospitalization

    # If the patient is initialized at time step 0, the remaining hours is drawn randomly from the original hours.
    hours_left = np.random.choice(list(np.arange(parameters.min_hospitalization, length_of_hospitalization + 1))) \
        if hour == 0 else length_of_hospitalization

    return hours_left


#######################################################################################################################
# SIMULATION
#######################################################################################################################

def simulation():
    """Runs the simulation after variables are set and agents initialized."""

    if parameters.print_to_console:
        print(u"\u2713\tThe main simulation has started.")
        parameters.progress(0, parameters.number_of_days * 24)

    ##############################

    # Iterates over each hour time step.
    for step, hour in enumerate(parameters.time_steps):
        # If no agents carry the virus, the simulation is stopped.
        if hour != parameters.t0 and parameters.carriers[hour - 1] == 0:
            if parameters.print_to_console:
                parameters.progress(parameters.number_of_days * 24, parameters.number_of_days * 24)
            return hour

        h = hour % 24

        # (1/5) Generates new shifts and admits patients if available beds.
        if h in parameters.new_shifts:
            for role, info in parameters.shifts.items():
                if h in list(info.keys()):
                    for w in range(parameters.number_of_wards):
                        chosen = get_new_shift(w + 1, role, h)
                        for staff in chosen:
                            staff.enter_ward(hour, info[h]['dur'], w + 1)
        for available_bed in range(sum(parameters.available_beds.values())):
            admit_patient(hour)

        # (2/5) Presence: Counts down hours of shift, hospitalization, resting, quarantine and/or isolation.
        parameters.shuffle_list_of_agents()
        for agent in parameters.agents_in_simulation:
            agent.tick_presence(hour)

        # (3/5) Generates contacts for the present agents in the ward.
        present = [a for a in parameters.agents_in_simulation if a.currently_present]
        generate_contacts(hour, present)

        # (4/5) Spreading: Spreads pathogens and counts down hours of incubation, infection and/or test waiting time.
        for agent in parameters.agents_in_simulation:
            if agent.carrier:
                agent.tick_spread(hour)

        # (5/5) Summarizes data and updates progress.
        parameters.summarize_time_step(hour)

        if parameters.print_to_console:
            parameters.progress(step, parameters.number_of_days * 24)

    return parameters.time_steps[-1]


############################################################

def get_new_shift(ward, role, hour):
    """
    Returns a list of staff assigned to a shift. Depends on whether the shift assignment is random or follows percentage
    of employment, and whether resting time is taken into account for the nurses.

    :param ward: Integer, ward number.
    :param role: String, One of the given staff roles ('NUR', 'MED' or 'ADM').
    :param hour: Integer, hour of the day (between 0 and 23).
    :return: chosen: List, containing the Staff-inheriting objects assigned for the given shift.
    """

    chosen = []
    required = parameters.shifts[role][hour]['num']

    # Available *regular staff* must not be in a quarantine or in isolation, must have the role of interest, belong to
    # the given ward, and lastly not already working in the ward.
    available = [agent for agent in parameters.agents_in_simulation
                 if agent.ward == ward
                 if agent.role == role
                 if agent.regular_staff
                 if agent.restriction['hours'] == 0
                 if not agent.currently_present]

    # There are enough staff, choice may depend on working percentage and remaining resting hours.
    if len(available) >= required:
        chosen = get_prioritized_staff(available, required, role)

    # There is a deficit of available regular staff and temporary substitutes are called.
    else:
        remaining = required - len(available)
        temporary_staff = [agent for agent in parameters.agents_in_simulation
                           if agent.ward == ward
                           if agent.role == role
                           if not agent.regular_staff
                           if agent.restriction['hours'] == 0
                           if not agent.currently_present]

        if len(available) + len(temporary_staff) >= required:
            chosen = available + get_prioritized_staff(temporary_staff, remaining, role)
        else:
            remaining -= len(temporary_staff)
            substitutes = []
            for i in range(remaining):
                substitute = hire_staff(ward=ward, role=role, hour=hour, regular_staff=False)
                substitutes.append(substitute)
            chosen = available + temporary_staff + substitutes

    if len(chosen) != required:
        print("stop en hal")
    return chosen


def generate_contacts(h, present):
    """
    Generates contact data, randomly or non-randomly, for the given hour based on present agents in the ward.
    Important: The parameter contact_randomization randomizes contacts within a ward.

    :param h: Integer, simulation hour + t0.
    :param present: List, containing all agents present in the ward.
    :return: None
    """

    # Draws number of contacts from a folded normal distribution with parameters based on the given hour.
    avg, std = parameters.contact[h % 24]['avg'], parameters.contact[h % 24]['std']
    number_per_bed = stats.norm.rvs(loc=avg, scale=std, size=None)
    if number_per_bed <= 0:
        parameters.number_of_contacts[h] = 0
        return

    number = int(round(number_per_bed * parameters.beds_per_ward * parameters.number_of_wards))
    number = int(round(number * (1 - parameters.social_distancing)))
    parameters.number_of_contacts[h] = number

    # Draws a set of agents constituting person A in the contact pairs for this hour.
    probability = np.array([a.extroversion for a in present])
    sum_probability = sum(probability)
    relative_probability = np.divide(probability, sum_probability)

    non_randomized_contacts = int(round(number * (1 - parameters.contact_randomization)))
    chosen = list(np.random.choice(present, non_randomized_contacts, p=relative_probability))

    if parameters.contact_randomization > 0:
        chosen = chosen + list(np.random.choice(present, number - len(chosen)))

    # Draws contact for all chosen individuals.
    for agent in set(chosen):
        role, ward = agent.role, agent.ward
        contacts_agent = chosen.count(agent)
        available = present.copy()
        available.remove(agent)

        # Draws a set of neighbors, person B, for all chosen agents.
        relative_combination = np.array([a.contact_data['comb'][ward][role] for a in available])

        sum_combination = sum(relative_combination)
        p_relative_combination = np.divide(relative_combination, sum_combination)
        non_randomized_contacts = int(round(contacts_agent * (1 - parameters.contact_randomization)))
        neighbors = list(np.random.choice(available, non_randomized_contacts, p=p_relative_combination))

        if parameters.contact_randomization > 0:
            relative_ward = np.array([a.contact_data['ward'][ward][role] for a in available])

            sum_ward = sum(relative_ward)
            p_relative_ward = np.divide(relative_ward, sum_ward)
            remaining_number = contacts_agent - len(neighbors)
            neighbors = neighbors + list(np.random.choice(available, remaining_number, p=p_relative_ward))

        # Adds a two-way contact between agent and neighbors.
        for neighbor in set(neighbors):
            contacts = neighbors.count(neighbor)
            agent.add_contact(neighbor, h, contacts)
            neighbor.add_contact(agent, h, contacts)

            if parameters.log_as_network:
                parameters.add_edge_to_network(agent, neighbor, contacts, h)


##############################


def get_relative_percentages(available):
    """
    Calculates relative probabilities of being assigned to a shift based on percentage of employments.

    :param available: List of available individuals suited for the given shift.
    :return: relative_percentages: List corresponding to 'available' where each entry represents a relative probability
             of being assigned to the given shift. The list entries sum to 1.0.
    """

    percentages = np.array([a.percentage for a in available])
    sum_percentages = sum(percentages)
    relative_percentages = np.divide(percentages, sum_percentages)

    return relative_percentages


def get_prioritized_staff(available, required, role):
    """
    Chooses staff from a role based on their percentage of employment (ALL) and remaining resting hours (NUR).

    :param available: List, containing available Staff-inheriting objects.
    :param required: Integer, number of required staff members.
    :param role: String, three letter code denoting role in the ward ('NUR', 'MED', 'ADM', 'PAT').
    :return: prioritized: List, containing the chosen staff members.
    """

    prioritized = []

    # MED, ADM and potentially NUR agents are chosen based on their percentage of employment.
    if role == 'MED' or role == 'ADM' or (role == 'NUR' and not parameters.resting_time):
        relative_percentages = get_relative_percentages(available)
        prioritized = list(np.random.choice(available, required, p=relative_percentages, replace=False))
        return prioritized

    # NUR agent shifts may prioritize based on resting hours between shifts.
    else:
        hours_left_to_rest = {}
        for nur in available:
            hours_left_to_rest.setdefault(nur.resting_hours, []).append(nur)

        nurses_left = required

        # Loops through remaining resting hours in ascending order, prioritizing nurses with lowest remaining hours.
        prioritized = []
        for hours_left, nurses in sorted(hours_left_to_rest.items()):
            if len(nurses) >= nurses_left:
                relative_percentages = get_relative_percentages(nurses)
                chosen = list(np.random.choice(nurses, nurses_left, p=relative_percentages, replace=False))
                prioritized.extend(chosen)
                nurses_left -= len(chosen)
                break
            else:
                prioritized.extend(nurses)
                nurses_left -= len(nurses)

    return prioritized


#######################################################################################################################
# SUMMARIZE
#######################################################################################################################

def summarize(folder, file):
    """
    Summarizes main model output. Key numbers are saved to a file, results may be plotted (parameters.plot_results) and
    the generated temporal contact network exported (parameters.log_as_network).

    :param folder: String, absolute path to desired output folder.
    :param file: String, desired file name of output (.txt) and network (.gexf) files.
    :return: cfr, ifr: Floats, case fatality rates based on confirmed and actual cases.
    """

    # Key numbers
    fatalities = len(parameters.fatalities)
    unique_cases = len(set(parameters.individual_cases))
    unique_carriers = len(set(parameters.individual_carriers))
    secondary_fatalities = len([k for k, v in parameters.fatalities.items() if v['cause'] == 'secondary'])

    # Case fatality rate
    cfr = 100 * np.divide(fatalities, unique_cases) if fatalities > 0 else 0.0
    ifr = 100 * np.divide(fatalities, unique_carriers) if fatalities > 0 else 0.0

    if parameters.print_to_console:
        print("-\tFatalities:\t\t\t{}\t({} from secondary infection)".format(fatalities, secondary_fatalities))
        print("-\tConfirmed cases:\t{}\t(CFR: {}%)".format(unique_cases, round(cfr, 2)))
        print("-\tActual cases:\t\t{}\t(IFR: {}%)".format(unique_carriers, round(ifr, 2)))
        print()

    ##############################

    # Write to file
    write_to_file(folder, file, cfr, ifr)

    # Plot key results
    if parameters.plot_results:
        plot_results(folder, file)

    # Export temporal network
    if parameters.log_as_network:
        export_networks(folder, file)


def write_to_file(folder, file, cfr_confirmed, cfr_carriers):
    """
    Writes model output to .txt file.

    :param folder: String, absolute path to desired output folder.
    :param file: String, desired file name of output (.txt) file.
    :param cfr_confirmed: Float, case fatality rate based on confirmed cases.
    :param cfr_carriers: Float, case fatality rate based on actual number of carriers.
    :return: None
    """

    if os.path.exists('{}{}.txt'.format(folder, file)):
        overwrite = input("This file already exists, do you want to overwrite it? (Y/N) ")
        if not overwrite:
            return

    with open("{}{}.txt".format(folder, file), "w") as f:
        f.write("cfr\t{}\n".format(cfr_confirmed))
        f.write("ifr\t{}\n".format(cfr_carriers))
        f.write("carriers\t{}\n".format(parameters.carriers))
        f.write("infected_all\t{}\n".format(parameters.infected_all))
        f.write("infected_confirmed\t{}\n".format(parameters.infected_confirmed))
        f.write("cumulative_cases\t{}\n".format(parameters.cumulative_cases))
        f.write("cumulative_deaths\t{}\n".format(parameters.cumulative_deaths))
        f.write("incubation\t{}\n".format(parameters.agents_incubation))
        f.write("asymptomatic\t{}\n".format(parameters.agents_asymptomatic))
        f.write("symptomatic\t{}\n".format(parameters.agents_symptomatic))
        f.write("temporary_staff\t{}\n".format(parameters.temporary_staff))
        f.write("quarantine\t{}\n".format(parameters.quarantine_agents))
        f.write("isolation\t{}\n".format(parameters.isolation_agents))
        f.write("contacts\t{}\n".format(parameters.number_of_contacts))


def export_networks(folder, file):
    """
    Exports the temporal contact network and the infection tree as two .gexf files in the given file location.

    :param folder: String, absolute path to desired output folder.
    :param file: String, desired file name of output (.gexf) file.
    :return: None
    """

    # Contact network
    exported = parameters.network
    for u, v, attributes in parameters.network.edges(data=True):
        list_of_contacts = attributes.get('listed_contacts')
        exported.add_edge(u, v, listed_contacts=str(list_of_contacts))
    nx.write_gexf(exported, '{}{}_contacts.gexf'.format(folder, file))

    # Infection tree
    nx.write_gexf(parameters.tree, '{}{}_tree.gexf'.format(folder, file))


def plot_results(folder, file):
    """
    Plots main results and saves them as a .png file.

    :param folder: String, absolute path to desired output folder.
    :param file: String, desired file name of output (.png) file.
    :return: None
    """

    fig, ax = plt.subplots(ncols=2, nrows=2)
    fig.set_size_inches(12, 8)
    x = [np.divide(h - parameters.t0, 24) for h in parameters.time_steps]

    # (1/4) Confirmed infected, infected and carriers
    color_map = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=4), cmap=mpl.cm.Reds)
    color_map.set_array([])
    c = [color_map.to_rgba(i + 1) for i in range(4)]

    ax.flat[0].plot(x, list(parameters.carriers.values()), label='carriers', color=c[0], alpha=0.8)
    ax.flat[0].plot(x, list(parameters.infected_all.values()), label='infected', color=c[1], alpha=0.8)
    ax.flat[0].plot(x, list(parameters.infected_confirmed.values()), label='confirmed', color=c[2], alpha=0.8)

    ax.flat[0].legend(frameon=False)
    ax.flat[0].set_xlabel('Simulation time (days)')
    ax.flat[0].set_ylabel('Number of individuals (-)')
    ax.flat[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.flat[0].yaxis.set_major_locator(MaxNLocator(integer=True))

    # (2/4) Cumulative confirmed cases and deaths
    cases = parameters.cumulative_cases[-1]
    deaths = parameters.cumulative_deaths[-1]
    cases_pos, deaths_pos = 0.90, round(np.divide(deaths, cases) + 0.08, 2)

    ax.flat[1].plot(x, list(parameters.cumulative_cases), label='confirmed cases', color=c[2])
    ax.flat[1].plot(x, list(parameters.cumulative_deaths), label='confirmed deaths', color=c[3])
    ax.flat[1].text(0.88, cases_pos, '{} cases'.format(cases), ha='center', transform=ax.flat[1].transAxes)
    ax.flat[1].text(0.88, deaths_pos, '{} deaths'.format(deaths), ha='center', transform=ax.flat[1].transAxes)

    ax.flat[1].legend(frameon=False)
    ax.flat[1].set_xlabel('Simulation time (days)')
    ax.flat[1].set_ylabel('Cumulative cases (-)')
    ax.flat[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.flat[1].yaxis.set_major_locator(MaxNLocator(integer=True))

    # (3/4) Incubation, asymptomatic and symptomatic
    colors = {'E': (np.divide(247, 255), np.divide(150, 255), np.divide(70, 255), 0.8),
              'I': (np.divide(192, 255), np.divide(80, 255), np.divide(77, 255), 0.8),
              'A': (np.divide(128, 255), np.divide(100, 255), np.divide(162, 255), 0.8)}

    ax.flat[2].plot(x, list(parameters.agents_incubation.values()), label='incubation', color=colors['E'])
    ax.flat[2].plot(x, list(parameters.agents_asymptomatic.values()), label='asymptomatic', color=colors['A'])
    ax.flat[2].plot(x, list(parameters.agents_symptomatic.values()), label='infection', color=colors['I'])

    ax.flat[2].legend(frameon=False)
    ax.flat[2].set_xlabel('Simulation time (days)')
    ax.flat[2].set_ylabel('Number of individuals (-)')
    ax.flat[2].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.flat[2].yaxis.set_major_locator(MaxNLocator(integer=True))

    # (4/4) Quarantine and isolation
    color_map = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=3), cmap=mpl.cm.BuPu)
    color_map.set_array([])
    c = [color_map.to_rgba(i + 1) for i in range(3)]

    ax.flat[3].plot(x, list(parameters.temporary_staff.values()), label='temporary staff', color=c[0], alpha=0.8)
    ax.flat[3].plot(x, list(parameters.quarantine_agents.values()), label='quarantine', color=c[1], alpha=0.8)
    ax.flat[3].plot(x, list(parameters.isolation_agents.values()), label='isolation', color=c[2], alpha=0.8)

    ax.flat[3].legend(frameon=False)
    ax.flat[3].set_xlabel('Simulation time (days)')
    ax.flat[3].set_ylabel('Number of individuals (-)')
    ax.flat[3].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.flat[3].yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig('{}{}_plot.png'.format(folder, file), dpi=300)

    ##############################


#######################################################################################################################
# MAIN
#######################################################################################################################

def simple_run():
    if parameters.measure_elapsed_time:
        start = time.time()

    if parameters.print_to_console:
        parameters.print_message("The simulation has started.")

    options()
    set_variables()
    initialize_agents()
    hour = simulation()

    message = "The simulation is finished after {} simulation days.".format(int(round(np.divide(hour, 24))))
    if parameters.measure_elapsed_time:
        elapsed_time = time.time() - start
        if parameters.print_to_console:
            parameters.print_message(message, elapsed_time)
    elif parameters.print_to_console:
        parameters.print_message(message)

    if not os.path.exists('output'):
        os.makedirs('output')
    summarize(folder=os.path.abspath(os.path.curdir) + '\\output\\', file='simulation_final')


def multiple_runs(replications):
    parameters.print_message("The simulation has started.")

    options()

    for r in range(replications):
        set_variables()
        initialize_agents()
        hour = simulation()
        print(u'\u2713\t Replication {}/{}'.format(r + 1, replications))
        summarize(folder=os.path.abspath(os.path.curdir) + '\\output\\', file='multiple_'.format(r))
    parameters.print_message("The simulation is finished.")


simple_run()
# multiple_runs(10)
