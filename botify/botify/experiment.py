from enum import Enum

import mmh3


class Treatment(Enum):
    C = 0
    T1 = 1
    T2 = 2
    T3 = 3
    T4 = 4
    T5 = 5
    T6 = 6
    T7 = 7
    T8 = 8
    T9 = 9


class Split(Enum):
    HALF_HALF = 2
    THREE_WAY = 3
    FOUR_WAY = 4
    FIVE_WAY = 5
    SEVEN_WAY = 7
    EIGHT_WAY = 8
    NINE_WAY = 9


class Experiment:
    """
    Represents a single A/B experiment. Assigns
    any user to one of the treatments based on
    experiment name and user ID.

    An example usage::

        experiment = Experiments.AA
        if experiment.assign(user) == Treatment.C:
            # do control actions
            ...
        elif experiment.assign(user) == Treatment.T1:
            # do treatment actions
            ...

    """

    def __init__(self, name: str, split: Split):
        self.name = name
        self.split = split
        self.hash = mmh3.hash(self.name)

    def assign(self, user: int) -> Treatment:
        user_hash = mmh3.hash(str(user), self.hash, False)
        return Treatment(user_hash % self.split.value)

    def __repr__(self):
        return f"{self.name}:{self.split}"


class Experiments:
    """
    A static container for all the existing experiments.
    """

    # TODO Семинар 1, Шаг 4 -  Запускаем A/B эксперимент.
    AA = Experiment("AA", Split.HALF_HALF)
    # TODO Семинар 1, Шаг 4.1 - Создаем эксперимент STICKY_ARTIST
    STICKY_ARTIST = Experiment("STICKY_ARTIST", Split.HALF_HALF)

    TOP_POP = Experiment("TOP_POP", Split.FOUR_WAY)
    USER_BASED = Experiment("USER_BASED", Split.HALF_HALF)
    PERSONALIZED = Experiment("PERSONALIZED", Split.THREE_WAY)
    DSSM = Experiment("DSSM", Split.HALF_HALF)
    CONTEXTUAL_DSSM_LFM = Experiment("CONTEXTUAL_DSSM_LFM", Split.FOUR_WAY)
    GCF = Experiment("GCF", Split.HALF_HALF)
    ALL = Experiment("ALL", Split.EIGHT_WAY)

    def __init__(self):
        self.experiments = [Experiments.STICKY_ARTIST]
