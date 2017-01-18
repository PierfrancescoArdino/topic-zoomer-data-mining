
# dataset = pd.read_csv('output_1.csv')


def read_dataset(filename):
	dataset = []
	file = open(filename, 'r')
	lines = list(file)
	lines = lines[1:]
	for line in lines:
		element = line.split(',', 2)
		element[0] = float(element[0])
		element[1] = float(element[1])
		dataset.append(element)
	return dataset
