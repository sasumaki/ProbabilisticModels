import pandas as pd


def genrate_transition_table(transition_string="bbbaabbaaabbabbaaaaabbab"
                             ):
    transition_probabilities = pd.DataFrame()
    transition_probabilities["X_t+1"] = ["a", "a", "b", "b"]
    transition_probabilities["X_t"] = ["a", "b", "a", "b"]

    transitions = len(transition_string)
    transition_tokenized = []
    aa = ab = ba = bb = 0
    for letter in transition_string:
        transition_tokenized.append(letter)
    prev_state = None
    for state in transition_tokenized:
        if(prev_state == None):
            prev_state = state
            continue
        else:
            if(state == "a"):
                if(prev_state == "a"):
                    aa += 1
                else:
                    ba += 1
            else:
                if(prev_state == "a"):
                    ab += 1
                else:
                    bb += 1

        prev_state = state

    # State transition probabilities counted from the evidence. Using laplace smoothing.
    transition_probabilities["P(X_t+1 | X_t)"] = [(aa + 1)/(transitions + 3),
                                                  (ab + 1)/(transitions + 3), (ba + 1)/(transitions + 3), (bb + 1)/(transitions + 3)]
    initial_state_probabilities = {"a": 3/4, "b": 1/4}
    return(transition_probabilities, initial_state_probabilities)


def generate_emission_table():
    emission_probabilities = pd.DataFrame(
        data={"E": ["A", "H", "A", "H"], "X": ["a", "a", "b", "b"]})
    # I felt lazy and calculated each by hand from the exercise data
    # Laplace smoothing
    probabilities = [(4 + 1) / (24 + 4), (8 + 1) / (24 + 4),
                     (7 + 1) / (24+4), (5+1)/(24+4)]
    emission_probabilities["P(E|X)"] = probabilities
    return(emission_probabilities)


def generate_hmm():
    transition_probabilities, initial_state_probabilities = genrate_transition_table()
    emission_probabilities = generate_emission_table()
    return(transition_probabilities, initial_state_probabilities, emission_probabilities)
