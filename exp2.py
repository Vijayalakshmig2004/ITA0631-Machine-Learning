import csv

def candidate_elimination(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)

    attributes = data[0][:-1]
    examples = data[1:]

    S = ['0'] * len(attributes)
    G = [['?'] * len(attributes)]

    for row in examples:
        instance, label = row[:-1], row[-1]
        if label.lower() == 'yes':
            for i in range(len(S)):
                if S[i] == '0':
                    S[i] = instance[i]
                elif S[i] != instance[i]:
                    S[i] = '?'
            G = [g for g in G if all(s == '?' or s == g[i] or g[i] == '?' for i, s in enumerate(S))]
        elif label.lower() == 'no':
            new_G = []
            for g in G:
                for i in range(len(g)):
                    if g[i] == '?':
                        if S[i] != '?':
                            new_hypothesis = g.copy()
                            new_hypothesis[i] = S[i]
                            if new_hypothesis not in new_G:
                                new_G.append(new_hypothesis)
            G = new_G

    print("Final Specific hypothesis S:", S)
    print("Final General hypotheses G:")
    for g in G:
        print(g)

candidate_elimination('cand_elim_dataset.csv')
