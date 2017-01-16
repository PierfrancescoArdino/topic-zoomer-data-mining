

#dataset = pd.read_csv('output_1.csv')
def read_dataset(filename):
	file = open(filename, 'r')

	dataset = []
	for line in file:
	    dataset.append(line.split(',',2))

	for i in range(len(dataset)):
		dataset[i][2]=dataset[i][2].replace("{","")
		dataset[i][2]=dataset[i][2].replace("}","")
		dataset[i][2]=dataset[i][2].split('|')

	return dataset