import csv

def find_s_algorithm(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)

    attributes = data[0][:-1]
    examples = data[1:]

    hypothesis = ['0'] * len(attributes)

    for row in examples:
        if row[-1].lower() == 'yes':
            for i in range(len(hypothesis)):
                if hypothesis[i] == '0':
                    hypothesis[i] = row[i]
                elif hypothesis[i] != row[i]:
                    hypothesis[i] = '?'

    print("Most specific hypothesis is:", hypothesis)

find_s_algorithm('finds_dataset.csv')
