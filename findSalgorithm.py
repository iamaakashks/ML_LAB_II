def find_s(dataset):
    hypothesis = dataset[0][: -1]
    for example in dataset:
        if example[-1] == "Yes":
            for i in range(len(hypothesis)):
                if example[i] != hypothesis[i]:
                    hypothesis[i] = "?"
    return hypothesis

dataset = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Warm', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'warm', 'Change', 'No']
]

hypothesis = find_s(dataset)
print("Hypothesis:", hypothesis)