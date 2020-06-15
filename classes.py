"""
This file covers all classes included in the epidemiological model.

Author: Signe Onstad Sævareid
Date: March 6th 2020
File: classes.py
-----------------------------------------------------------------------------------------------------------------------
* Imported modules: typing (type distinction, https://docs.python.org/3/library/typing.html)
                    numpy (scientific computing, version 1.18.1, https://numpy.org/)

* Imported files: parameters.py

* Class diagram:
    - Individual (abstract)
        - Staff (abstract)
            - HealthcareWorker (abstract)
                - NUR
                - MED
            - ADM
        - PAT
-----------------------------------------------------------------------------------------------------------------------
"""

from typing import Dict
import numpy as np

import parameters


#######################################################################################################################


class Individual:
    """
    The Individual class has two direct subclasses: Staff and PAT.

    :param: id_number: Integer, unique number following the Organism object throughout the simulation.
    :param: role: String, three letter code denoting role in the ward ('NUR', 'MED', 'ADM', 'PAT').
    :param: ward: Integer, number indicating to which ward the Individual belongs.
    :param: extroversion: Float, average number of inter-individual contacts per hour per other individuals present.

    :var: incubation_period: Dictionary, containing log-normal parameters ('mu', 'sigma') for the incubation period.
    :var: t-presymptomatic: Float, number of days before infection onset where the individual becomes infectious.
    :var: p_susceptible: Float, probability of *not* acquiring immunity when recovering from an infection.
    :var: t_A: Dictionary, containing log-normal parameters ('mu', 'sigma') for duration of an asymptomatic infection.

    :var: t_notify_contacts: Integer, number of hours before symptom onset where contacts must be notified.
    :var: t_quarantine: Integer, number of quarantine days from potential exposure.
    :var: t_isolation: Integer, number of isolation days from when symptoms are gone.
    :var: t_milder: Integer, Number of quarantine days after symptoms of a milder respiratory infections have gone.
    :var: p_remember: Float, probability of remembering a unique individual of contact during t_notify_contacts.

    :var: p_V: Float, probability of being vaccinated.
    :var: p_E: Float, probability of vaccine providing full immunity.
    """

    incubation_period: Dict[str, str]
    t_presymptomatic: float
    p_susceptible: float
    t_A: Dict[str, str]

    t_notify_contacts: int
    t_quarantine: int
    t_isolation: int
    t_milder: int
    p_remember: float

    p_V: float
    p_E: float

    # Overridden in Staff and PAT
    p_A: float
    t_I: Dict[float, float]
    p_D: float

    ############################################################

    def __init__(self, id_number, role, ward, extroversion):
        self.id_number = id_number
        self.role = role
        self.ward = ward
        self.extroversion = extroversion

        self.alive = True                   # Boolean, tells if the Individual is alive or not.
        self.course_of_disease = []         # List, holding tuples describing course of disease.

        self.currently_present = False      # Boolean, tells if the Individual is present in the ward or not.
        self.hours_left_in_ward = 0         # Integer, hours left in the ward be it shift or hospitalization.
        self.contact_data = {}              # Dictionary, holding data on contact rates with roles and wards.
        self.contacts = {}                  # Dictionary, for holding contact information per hour.

        self.susceptible = True             # Boolean, tells if the Individual is susceptible or not.
        self.positive_test = False  # Boolean, tells if the individual has tested positive for the virus.
        self.carrier = False
        self.infectious = 'N'               # String, level of infectiousness ('N' < 'A' < 'I')

        self.restriction = {'level': 'N',   # String, level of contact restriction ('N' < 'Q' < 'I')
                            'hours': 0,     # String, reason for level of restriction.
                            'reason': ""}   # Integer, hours left in either quarantine or isolation.

        self.incubation = 0                 # Integer, remaining hours of the incubation period.
        self.infection = 0                  # Integer, remaining hours of the infection (symptomatic or asymptomatic).
        self.symptoms = {}                  # Dictionary, holding hour of symptom onset and list of symptoms.

        self.fatal_secondary = 0            #

        self.test_results = {}              # Dictionary, holding remaining waiting time and test result.
        self.wait_for_test = 0              # Integer, number of hours before being tested while having mild symptoms.
        self.R = None                       # Integer, counts the number of individuals transmitted by this Individual.

        self.set_contact_data()

        if self.p_V > 0:                    # If there is vaccine, the individual may obtain immunity.
            self.vaccinate()

    def __repr__(self):
        return '{}{} (ward {})'.format(self.role, self.id_number, self.ward)

    ############################################################
    def set_contact_data(self):
        """Calculates relative contact rate data to other agents depending on ward and role."""

        contact_data = {'ward': {}, 'comb': {}}

        contact_rate_roles = parameters.contact_rate_roles
        cooperation = 1 - parameters.ward_isolation
        extroversion = self.extroversion
        agent_role = self.role
        agent_ward = self.ward

        for combination in ['ward', 'comb']:
            for ward in np.arange(1, parameters.number_of_wards + 1):
                contact_data[combination][ward] = {}
                for role in ['NUR', 'MED', 'ADM', 'PAT']:
                    if ward == agent_ward:
                        w = 1
                        c = round(contact_rate_roles[role][agent_role] * extroversion, 10)
                    elif not (agent_role == 'PAT' or role == 'PAT'):
                        w = cooperation
                        c = round(contact_rate_roles[role][agent_role] * extroversion, 10)
                    else:
                        w = 0
                        c = 0
                    contact_data[combination][ward][role] = w if combination == 'ward' else c * w

        self.contact_data = contact_data

    def vaccinate(self):
        """The individual is no longer susceptible to the infection if the vaccine provides full immunity."""

        p_immunity = self.p_V * self.p_E

        if np.random.binomial(1, p=p_immunity):
            self.susceptible = False
            self.course_of_disease.append(('R', 0))

    ############################################################

    def tick_presence(self, hour):
        """Counts down hours of presence in the ward and facilitates leaving when there is no hours left."""

        # Hours left in the ward are counted down. When reaching 0, the agent leaves the ward.
        if self.currently_present:
            self.hours_left_in_ward -= 1
            if self.hours_left_in_ward == 0:
                self.leave_ward(hour)

        # Hours left in quarantine or isolation are counted down. When reaching 0, the agent returns to no restrictions.
        if self.restriction['hours'] > 0:
            self.restriction['hours'] -= 1
            if self.restriction['hours'] == 0:
                self.restriction['level'] = 'N'
                self.restriction['reason'] = ""

    def leave_ward(self, hour, forced=False):
        """
        An agent may leave the ward by discharge or when ending a shift, or being forced due to quarantine or isolation.

        :param hour: Integer, day hour marking the time point where the agent leaves the ward.
        :param forced: Boolean, tells if the person leaves the ward due to restrictions or not.
        :return: None
        """

        raise NotImplementedError

    def add_contact(self, neighbor, hour, number):
        """
        Adds the number of contacts with the given neighbor for the given hour.

        :param neighbor: Individual subclass, i.e. object of 'NUR', 'MED', 'ADM' or 'PAT'  # TODO: How to declare type?
        :param hour: Integer, day hour marking the time point of contact.
        :param number: Integer, number of contacts made between self and neighbor in the given hour.
        :return: None
        """

        # Add the hour as a key if it does not exist.
        if hour not in self.contacts:
            self.contacts[hour] = {}

        # Add the number of contacts if the neighbor does not exist as a key for the given hour.
        if neighbor not in self.contacts[hour]:
            self.contacts[hour][neighbor] = number

        # Add the existing registered contacts between self and neighbor with the additional number of contacts.
        else:
            old_number = self.contacts[hour][neighbor]
            self.contacts[hour][neighbor] = old_number + number

    ##############################

    def tick_spread(self, hour):
        """Count downs hours of incubation (tE) and infection period (tA or tI)."""
        # If present in the ward and infectious, the agent can infect other agents through contact.
        if self.currently_present and self.infectious is not 'N':
            """# TODO REMOVE
            if not self.carrier:
                print(self)"""

            neighbors = self.contacts.get(hour)
            if neighbors is not None:
                self.infect_neighbors(neighbors, hour)

        # The incubation period (E) is counted down. When reaching the presymptomatic infectious period, the level of
        # infectiousness is set to 'A'. When the incubation period is over, the agent enters an infection.
        if self.incubation > 0:
            self.incubation -= 1
            if self.incubation == self.t_presymptomatic:
                self.infectious = 'A'
            if self.incubation == 0:
                self.enter_infection(hour)

        # The infection period (A or I) is counted down. When the infection period is over, the infection ends by
        # recovery or a fatal outcome.
        if self.infection > 0:
            self.infection -= 1
            if self.infection == 0:
                self.end_infection(hour)

        # If the agent has been tested for COVID-19, the waiting time for the results is counted down. In case of a
        # positive test, the agent enters isolation and must notify all close contacts who may be at risk.
        if self.test_results:
            self.test_results['remaining_waiting_time'] -= 1
            if self.test_results['remaining_waiting_time'] == 0 and self.test_results['positive']:
                self.enter_restriction(hour, "Positive test", 'I', self.test_results['remaining_restriction_time'])
                self.notify_close_contacts(hour)
                self.positive_test = True
                self.test_results = {}

                parameters.individual_cases.append(self.id_number)

        # If having a milder infection, the agent is tested 48 hours after symptom onset if still having symptoms.
        if self.wait_for_test > 0:
            self.wait_for_test -= 1
            if self.wait_for_test == 0 and self.infection > 0:
                self.conduct_test()

    def infect_neighbors(self, neighbors, hour):
        """
        Calculates transmission probability based on level of infectiousness (alpha), restriction level (beta), relative
        infectivity to influenza virus (gamma) and number of contacts in the course of the given hours (c).

        :param neighbors: List, contains all agents with whom the infectious person has has contact with.
        :param hour: Integer, day hour marking the time point of potential transmission.
        :return: None
        """

        # Transmission parameters constant with respect to neighbor.
        alpha = parameters.alpha[self.infectious]
        gamma = parameters.gamma

        # Loops through all susceptible neighbors and draws transmission from a Bernoulli distribution.
        infected_neighbors = []

        for neighbor in neighbors:
            if neighbor.susceptible:
                beta = min(parameters.beta[self.restriction['level']], parameters.beta[neighbor.restriction['level']])
                c = self.contacts[hour][neighbor]
                p_t = 1 - np.power(1 - alpha * beta * gamma * 0.003, c)
                transmission = np.random.binomial(1, p=p_t)

                if transmission == 1:
                    infected_neighbors.append(neighbor)

        for neighbor in infected_neighbors:
            if neighbor.susceptible:
                neighbor.catch_virus(hour)
                self.R[neighbor.role] += 1

                if parameters.log_as_network:
                    parameters.add_infection_route_to_tree(self, neighbor, hour)

    def catch_virus(self, hour):
        """If the individual is susceptible, an incubation period is initiated."""

        # When catching the virus, the agent is no longer susceptible. The carrier status is set to True, and the
        # counter for number of agents this agent has infected is set to start at zero.

        self.course_of_disease.append(('E', hour))
        self.susceptible = False
        self.carrier = True
        self.R = {'NUR': 0, 'MED': 0, 'ADM': 0, 'PAT': 0}

        parameters.individual_carriers.append(self.id_number)

        # Draws incubation time duration from a log-normal distribution. If the remaining incubation period is shorter
        # than the timespan of presymptomatic infectiousness, the agent is set to be infectious immediately.
        mu, sigma = self.incubation_period['mu'], self.incubation_period['sigma']
        self.incubation = int(round(24 * np.random.lognormal(mu, sigma, size=None)))
        if self.incubation <= self.t_presymptomatic:
            self.infectious = 'A'

    def enter_infection(self, hour):
        """Starts an infection, either asymptomatic or symptomatic."""

        # In case of an asymptomatic infection, the agent enters the infection without any further notice.
        if np.random.binomial(1, p=self.p_A):
            mu, sigma = self.t_A['mu'], self.t_A['sigma']
            self.infection = int(round(24 * np.random.lognormal(mu, sigma, size=None)))
            self.course_of_disease.append(('A', hour))

        # In case of a symptomatic infection, further act depends on whether the symptoms indicate COVID-19 or not.
        else:
            self.infectious = 'I'
            mu, sigma = self.t_I['mu'], self.t_I['sigma']
            self.infection = int(round(24 * np.random.lognormal(mu, sigma, size=None)))
            self.course_of_disease.append(('I', hour))
            self.fatal_secondary = self.fatal_secondary_infection()

            # Key symptoms accompanying the acute respiratory infection.
            assume_covid = self.get_symptoms(hour)
            self.handle_infection(hour, assume_covid)

    def end_infection(self, hour):
        """Ends an infection, either by a fatal outcome or by recovery."""

        # A fatal outcome requires a symptomatic course of disease and a positive value from a Bernoulli distribution.
        if self.infectious == 'I' and np.random.binomial(1, p=self.p_D):
            self.pass_away(hour, 'primary')

        elif self.fatal_secondary:
            self.pass_away(hour, 'secondary')

        # The Individual may return to a susceptible state if it does not obtain immunity.
        elif not np.random.binomial(1, p=(1 - self.p_susceptible)):
            self.susceptible = True
            self.course_of_disease.append(('S', hour))

        # If not a fatal outcome, the individual recovers.
        else:
            self.course_of_disease.append(('R', hour))

        # The Individual is no longer infectious.
        self.infectious = 'N'
        self.carrier = False

    def get_symptoms(self, hour):
        """
        Draws symptoms from three key symptoms that must be present in order to be tested for SARS-CoV-2.

        :param hour: Integer, day hour marking the time point of symptom onset.
        :return: assume_covid: Boolean, tells whether the symptoms give reason to believe COVID-19 or not.
        """

        # Draws 0 to 3 of the three symptoms using a Bernoulli trial with associated probabilities.
        symptom, p_symptoms = list(parameters.symptoms.keys()), list(parameters.symptoms.values())
        list_of_symptoms = [i for (i, v) in zip(symptom, np.random.binomial(1, p=p_symptoms)) if v]
        self.symptoms = {'onset': hour, 'symptoms': list_of_symptoms}

        # COVID-19 is only assumed if one of the three key symptoms are present.
        assume_covid = True if len(list_of_symptoms) > 0 else False

        return assume_covid

    def handle_infection(self, hour, assume_covid):
        """
        In case of a suspected COVID-19 infection, the person is tested and set in a quarantine until test results are
        available. In case of a milder respiratory infection, the quarantine lasts until 1 day after symptoms are gone.

        :param hour:  Integer, day hour marking the time point of infection.
        :param assume_covid: Boolean, tells if the person has symptoms indicating a COVID-19 infection or not.
        """

        if assume_covid:
            self.conduct_test()

            reason = "Suspected COVID-19 infection"
            duration = parameters.test_results_waiting_time
            milder_symptoms = False

        else:
            reason = "Mild respiratory infection"
            duration = self.infection + (24 * parameters.t_milder)
            milder_symptoms = True

        # Only individuals with no previous restrictions or those in close contact quarantines are set in quarantine.
        if self.restriction['level'] == 'N' or self.restriction['reason'] == 'Close contact with COVID-19 infected individual':
            self.enter_restriction(hour, reason, 'Q', duration, milder_symptoms)

    def conduct_test(self):
        """
        Tests an individual for SARS-CoV-2. Sets waiting time for the test result and whether the result will be
        positive or not. Assumes the test accuracy to be 100% and that incubation period yields a negative test.
        """

        waiting_time = parameters.test_results_waiting_time
        positive_test = True if self.infection > 0 else False

        remaining_restriction_time = self.infection + (24 * self.t_isolation) - parameters.test_results_waiting_time \
            if positive_test else 0

        self.test_results = {'remaining_waiting_time': waiting_time, 'positive': positive_test,
                             'remaining_restriction_time': remaining_restriction_time}

    def notify_close_contacts(self, hour):
        """
        In case of a positive test, (ideally all) individuals who have had contact with the infected individual in a
        given time interval before symptom onset should be contacted and set in a quarantine. The number of individuals
        set in a quarantine depends on 1) how many of them the infected person remembers and 2) the number of contacts
        required to be considered a close contact. Discharged patients are not contacted.

        :param: hour: Integer, day hour marking hour of positive test results.
        :return: None
        """

        # The critical hour interval is defined as t_notify_contacts hours preceding symptom onset.
        onset = self.symptoms['onset']
        hour_interval = list(np.arange(onset - parameters.t_notify_contacts, onset))

        # The set of individuals, number of contacts and last exposure if found by looping through the hour interval.
        close_contacts = {}
        for hour in hour_interval:
            if hour in self.contacts:
                for individual, contacts in self.contacts[hour].items():
                    if individual.role == 'PAT' and not individual.currently_present:
                        continue

                    if individual not in close_contacts:
                        close_contacts[individual] = {'contacts': contacts, 'last_exposure': hour}
                    else:
                        close_contacts[individual]['contacts'] = close_contacts[individual]['contacts'] + contacts
                        close_contacts[individual]['last_exposure'] = hour

        # A proportion (p_remember) of the unique individuals are remembered.
        unique_individuals = list(close_contacts.keys())
        number_of_remembered_individuals = int(round(len(unique_individuals) * parameters.p_remember))
        neighbors_to_notify = np.random.choice(unique_individuals, number_of_remembered_individuals, replace=False)

        # Loops through all remembered close contact neighbors if the number of contacts qualifies for quarantine.
        for neighbor in neighbors_to_notify:

            # If the neighbor is already in a quarantine or isolation, there is no need to contact them.
            if neighbor.restriction['level'] is not 'N':
                continue

            if close_contacts[neighbor]['contacts'] >= parameters.close_contact_definition:
                hours_since_last_exposure = hour - close_contacts[neighbor]['last_exposure']
                duration = (24 * self.t_quarantine) - hours_since_last_exposure + 1
                neighbor.enter_restriction(hour, "Close contact with COVID-19 infected individual", 'Q', duration)

    def pass_away(self, hour, cause):
        """
        When an agent passes away, the time of death is registered and the agent leaves the ward.

        :param hour: Integer, time point of death.
        :param cause: String, primary or secondary infection.
        :return: None
        """

        # The death is registered.
        self.alive = False
        self.course_of_disease.append(('D', hour))
        parameters.register_new_death(self.id_number, hour, self.role, cause)

        # Agent attribute values are reset in order to avoid unnecessary looping.
        self.carrier = False
        self.hours_left_in_ward = 0
        self.infectious = 'N'
        self.restriction['level'] = 'N'
        self.restriction['hours'] = 0

        if self.currently_present:
            self.leave_ward(hour, forced=True)

    def enter_restriction(self, hour, reason, restriction_level, duration, mild_symptoms=False):
        """
        Starts a quarantine or isolation for an Individual lasting for a given duration.

        :param hour: Integer, hour marking the time point of restriction.
        :param reason: String, reason for the increased restriction level.
        :param restriction_level: String, level of restriction ('Q' or 'I').
        :param duration: Integer, remaining hours of this restriction level.
        :param mild_symptoms: Boolean, set to True if the Individual enters restriction with mild respiratory symptoms.
        :return: None
        """

        raise NotImplementedError

    def fatal_secondary_infection(self):
        return


class Staff(Individual):
    """
    The Staff class inherits directly from Individual and has two direct subclasses: HealthcareWorker and ADM.

    :param: percentage: Float, percentage of employment.
    :param: regular_staff: Boolean, marks the difference between the regular staff and temporary substitutes.

    :var: p_A: Boolean, probability of an asymptomatic course of disease.
    :var: t_I: Dictionary, containing log-normal parameters ('mu', 'sigma') for drawing the duration of infection.
    :var: p_D: Boolean, probability of a fatal outcome.
    """

    p_A: float
    t_I: Dict[float, float]
    p_D: float

    ############################################################

    def __init__(self, id_number, role, ward, extroversion, percentage, regular_staff):
        Individual.__init__(self, id_number, role, ward, extroversion)
        self.percentage = percentage
        self.regular_staff = regular_staff
        # self.original_ward = ward

        self.working_hours = []     # Empty list for logging working hours.

    ############################################################

    def enter_ward(self, hour, duration, ward):
        """
        Called whenever a Staff member enters the ward when beginning a new shift.

        :param hour: Integer, day hour marking the start of the shift.
        :param duration: Integer, number of hours in the shift.
        :param ward: Integer, ward the agent enters.
        :return: None
        """

        self.currently_present = True
        self.hours_left_in_ward = duration
        self.working_hours.append(list(np.arange(hour, hour + duration)))

    def enter_restriction(self, hour, reason, restriction_level, duration, mild_symptoms=False):
        """Overrides enter_restriction() in Individual: Includes opportunity to send currently working Staff home."""

        self.restriction = {'level': restriction_level, 'hours': duration, 'reason': reason}

        # Staff currently working in the ward can be sent home directly.
        if parameters.send_staff_directly_home and self.currently_present:
            self.leave_ward(hour, forced=True)

    def leave_ward(self, hour, forced=False):
        """Overrides leave_ward() in Individual: Includes the opportunity to send agents directly home."""

        self.currently_present = False

        # In case of leaving the ward due to restrictions, the remaining hours left in the ward is set to zero.
        if forced and self.hours_left_in_ward > 0:
            self.hours_left_in_ward = 0


class HealthcareWorker(Staff):
    """
    The HealthcareWorker class inherits directly from Staff and has two direct subclasses: NUR and MED.
    """

    t_quarantine: int

    ############################################################

    def __init__(self, id_number, role, ward, extroversion, percentage, regular_staff):
        Staff.__init__(self, id_number, role, ward, extroversion, percentage, regular_staff)

    ############################################################

    def enter_restriction(self, hour, reason, restriction_level, duration, mild_symptoms=False):
        """Overrides enter_restriction() in Staff: Includes a future evaluation of testing if symptoms persist."""

        self.restriction = {'level': restriction_level, 'hours': duration, 'reason': reason}

        # Staff currently working in the ward can be sent home directly.
        if parameters.send_staff_directly_home and self.currently_present:
            self.leave_ward(hour, forced=True)

        # Starts countdown for taking test if symptoms still persist after 48 hours.
        if mild_symptoms:
            self.wait_for_test = 48


#######################################################################################################################


class ADM(Staff):
    """
    The ADM class inherits directly from Staff.
    """

    ############################################################

    def __init__(self, id_number, ward, extroversion, percentage, regular_staff):
        Staff.__init__(self, id_number, type(self).__name__, ward, extroversion, percentage, regular_staff)

    ############################################################


class MED(HealthcareWorker):
    """
    The MED class inherits directly from HealthcareWorker.
    """

    ############################################################

    def __init__(self, id_number, ward, extroversion, percentage, regular_staff):
        HealthcareWorker.__init__(self, id_number, type(self).__name__, ward, extroversion, percentage, regular_staff)

    ############################################################


class NUR(HealthcareWorker):
    """
    The NUR class inherits directly from HealthcareWorker.
    """

    ############################################################

    def __init__(self, id_number, ward, extroversion, percentage, regular_staff):
        HealthcareWorker.__init__(self, id_number, type(self).__name__, ward, extroversion, percentage, regular_staff)

        self.resting_hours = 0      # Integer, number of hours left before fully rested after a shift.

    ############################################################

    def tick_presence(self, hour):
        """Overrides tick_contact() in Individual: Includes countdown of resting hours for NUR currently not present."""

        if self.currently_present:
            self.hours_left_in_ward -= 1
            if self.hours_left_in_ward == 0:
                self.leave_ward(hour)

        if self.restriction['hours'] > 0:
            self.restriction['hours'] -= 1
            if self.restriction['hours'] == 0:
                self.restriction['level'] = 'N'
                self.restriction['reason'] = ""

        # If the agent is not present in the ward, potential remaining resting hours are counted down.
        else:
            if self.resting_hours > 0:
                self.resting_hours -= 1

    def leave_ward(self, hour, forced=False):
        """Overrides leave_ward() in Individual: Sets resting hours after an ended shift, resets if forced to leave."""

        self.currently_present = False

        if forced and self.hours_left_in_ward > 0:
            self.hours_left_in_ward = 0
            self.resting_hours = 0

        # When leaving the ward after an ended shift, the agent starts a resting period.
        if not forced:
            self.resting_hours = parameters.rest if isinstance(parameters.rest, int) else \
                int(round(np.random.lognormal(parameters.rest['mu'], parameters.rest['sigma'], size=None)))


class PAT(Individual):
    """
    The PAT class inherits directly from Individual.

    :param: hours_left_in_ward: Integer, hours left of hospitalization.
    :param: admission: Integer, simulation hour of patient admission.

    :var: p_A: Boolean, probability of an asymptomatic course of disease.
    :var: t_I: Dict, containing log-normal parameters ('mu', 'sigma') for drawing the duration of infection.
    :var: p_D: Boolean, probability of a fatal outcome.
    """

    p_A: float
    t_I: Dict[float, float]
    p_D: float

    p_secondary: float
    p_antibiotics: float
    p_resistant: float
    p_D_sensitive: float
    p_D_resistant: float

    ############################################################

    def __init__(self, id_number, ward, extroversion, admission, hours_left_in_ward):
        Individual.__init__(self, id_number, type(self).__name__, ward, extroversion)

        self.admission = admission
        self.hours_left_in_ward = hours_left_in_ward

        self.currently_present = True   # Boolean, A PAT object is only instantiated when entering the ward.
        self.discharge = None           # Integer, for registration of patient discharge.

    ############################################################

    def enter_restriction(self, hour, reason, restriction_level, duration, mild_symptoms=False):
        """Overrides enter_restriction() in Individual: The hospitalization duration is potentially extended."""

        self.restriction = {'level': restriction_level, 'hours': duration, 'reason': reason}

        # The patient either stays in the ward until planned treatment is finished or to restriction ceases.
        remaining_hours_in_the_ward = max(self.restriction['hours'], self.hours_left_in_ward)
        self.hours_left_in_ward = remaining_hours_in_the_ward + 1

    def leave_ward(self, hour, forced=False):
        """Overrides leave_ward() in Individual: Includes freeing up a bed in the ward and registers discharge."""

        self.currently_present = False

        # The bed in the ward is freed up.
        parameters.update_available_beds(self.ward, 1)
        self.discharge = hour


    def fatal_secondary_infection(self):
        """
        # Determines whether the agent will die from a secondary infection or not.

        :return: fatal_secondary: Boolean, determine if the agent dies from a secondary infection.
        """
        fatal_secondary = 0

        secondary_infection = np.random.binomial(1, p=self.p_secondary)
        if secondary_infection:
            antibiotics = np.random.binomial(1, p=self.p_antibiotics)
            if antibiotics:
                resistant = np.random.binomial(1, p=self.p_resistant)
                if resistant:
                    fatal_secondary = np.random.binomial(1, p=self.p_D_resistant)
                else:
                    fatal_secondary = np.random.binomial(1, p=self.p_D_sensitive)
            else:
                fatal_secondary = np.random.binomial(1, p=self.p_D_resistant)

        return fatal_secondary
