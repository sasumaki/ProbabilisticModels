"""
A protein is composed of a sequence of amino acids, which is often called
primary structure. While there are about twenty commonly occuring
amino acids, and they can be grouped many different ways, we will consider
them all to belong to either the Aliphatic or Hydroxyl group (to
simplify calculations).
A key factor in understanding the function of a particular protein is, however,
the secondary structure, that is, how the linked amino acids fold in
three dimensional space.
Two of the most important features of the seconary
structure are α helices and β sheets. However, it is difficult to assess the
secondary structure without expensive experimental techniques such as
crystallography. On the other hand, finding the sequence of amino acids
in a protein sequence is comparatively inexpensive. Consequently, a goal
of computational biology is to infer the secondary structure of a protein
given its primary structure (the sequence of amino acids).
Suppose we conduct a set of costly crystallography experiments and discover
the following secondary structures for the associated amino acid
sequences.
• βββααββ: HAAHHHA
• αααββ: AHHAH
• αββααα: HAHHAA
• ααββαβ: HHAAAH
Notation: α means α-helix, β means β sheet, A means aliphatic and
H means hydroxyl, and the i-th secondary structure element in the sequence
corresponds to the i-th amino acid
"""
import pandas as pd
import hmm
import matplotlib.pyplot as plt


def forward(acid_seq, initial_prob, transition, emission):
    # forward
    states = initial_prob.keys()
    forward = []
    f_prev = {}
    for i, observation in enumerate(acid_seq):
        curr = {}
        for state in states:
            if(i == 0):
                prev_sum = initial_prob[state]
            else:
                prev_sum = sum(f_prev[k]*float(transition[(transition["X_t"] == str(k))
                                                          & (transition["X_t+1"] == str(state))]["P(X_t+1 | X_t)"]) for k in states)
            em_prob = float(emission[(emission["E"] == str(
                observation)) & (emission["X"] == str(state))]["P(E|X)"])
            curr[state] = em_prob * prev_sum
        forward.append(curr)
        f_prev = curr

    return(forward)


def normalize(probabilities):
    normalized = []
    for probs in probabilities:
        a, b = probs.values()
        summed = a+b
        a = a/summed
        b = b/summed
        normalized.append({"a": a, "b": b})
    return(normalized)


def backward(acid_seq, initial_prob, transition, emission):
    states = initial_prob.keys()
    end_state = (acid_seq[-1])
    bkw = []
    b_prev = {}
    for i, observation_i_plus in enumerate(reversed(acid_seq)):
        b_curr = {}
        for state in states:
            if i == 0:

                b_curr[state] = 1
            else:

                b_curr[state] = sum(float(transition[(transition["X_t"] == str(state)) & (transition["X_t+1"] == l)]["P(X_t+1 | X_t)"]) * float(
                    emission[(emission["E"] == str(observation_i_plus)) & (emission["X"] == str(l))]["P(E|X)"]) * b_prev[l] for l in states)

        bkw.insert(0, b_curr)
        b_prev = b_curr
    return(bkw)


def forw_backw(forward, backward):
    posterior = []
    for f, b in zip(forward, backward):
        smoothed_a = f["a"] * b["a"]
        smoothed_b = f["b"] * b["b"]
        posterior.append({"a": smoothed_a, "b": smoothed_b})

    return(posterior)


def viterbi(acid_seq, transition, initial, emission):
    viterbi_sequence = ""
    T = [{}]
    states = initial.keys()
    for state in states:
        prob = float((emission[(emission["E"] == str(
            acid_seq[0])) & (emission["X"] == str(state))]["P(E|X)"]) * initial.get(str(state)))
        # print(prob)
        T[0][state] = {"probability": prob, "previous": None}
    for i, obs in enumerate(acid_seq):
        T.append({})
        i += 1
        for state in states:
            transitions = []
            for prev_st in states:
                transitions.append(T[i-1][str(prev_st)]["probability"] * float(transition[(
                    transition["X_t"] == str(prev_st)) & (transition["X_t+1"] == str(state))]["P(X_t+1 | X_t)"]))
            maximum_transition = max(transitions)
            for prev_st in states:
                if (T[i-1][str(prev_st)]["probability"] * float(transition[(transition["X_t"] == str(prev_st)) & (transition["X_t+1"] == str(state))]["P(X_t+1 | X_t)"])) == (maximum_transition):
                    maximum_probability = float((emission[(emission["E"] == str(
                        obs)) & (emission["X"] == str(state))]["P(E|X)"]) * maximum_transition)
                    T[i][state] = {
                        "probability": maximum_probability, "previous": prev_st}
                    break
    """
    This next part is almost straight from wikipedia article of Viterbi algorithm
    """
    opt = []
    max_prob = max(value["probability"] for value in T[-1].values())
    previous = None
    for st, data in T[-1].items():
        if data["probability"] == max_prob:
            opt.append(st)
            previous = st
            break
    for t in range(len(T) - 2, -1, -1):
        opt.insert(0, T[t + 1][previous]["previous"])
        previous = T[t + 1][previous]["previous"]

    viterbi_sequence = ('The steps of states are ', ' '.join(opt))

    return(viterbi_sequence)


if __name__ == '__main__':
    transition, initial, emission = hmm.generate_hmm()
    print(transition)
    print(emission)
    print("initial transition states: ", initial)
    acid_seq = ["HAHAHA", "HAAAHH"]
    forward_probabilities1 = forward(
        acid_seq[0], initial, transition, emission)
    forward_probabilities2 = forward(
        acid_seq[1], initial, transition, emission)
    print("Forward: ")
    print(forward_probabilities1)
    print(forward_probabilities2)

    backward_probabilities1 = backward(
        acid_seq[0], initial, transition, emission)
    backward_probabilities2 = backward(
        acid_seq[1], initial, transition, emission)
    print("Backwards: ")
    print(backward_probabilities1)
    print(backward_probabilities2)

    post1 = forw_backw(forward_probabilities1, backward_probabilities1)
    post1 = normalize(post1)
    post2 = forw_backw(forward_probabilities2, backward_probabilities2)
    post2 = normalize(post2)
    print("Posteriors: ")
    print(post1)
    print(post2)

    viterbi1 = viterbi(acid_seq[0], transition, initial, emission)
    viterbi2 = viterbi(acid_seq[1], transition, initial, emission)
    print(viterbi1)
    print(viterbi2)
