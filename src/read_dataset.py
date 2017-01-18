

#dataset = pd.read_csv('output_1.csv')
def read_dataset(filename):
	dataset = []
	for name in filename:
		file = open(name, 'r')
		
		for line in file:
		    dataset.append(line.split(',',2))

	for i in range(len(dataset)):
		dataset[i][2]=dataset[i][2].replace("{","")
		dataset[i][2]=dataset[i][2].replace("}","")
		dataset[i][2]=dataset[i][2].split('|')
		dataset[i][0]=float(dataset[i][0])
		dataset[i][1]=float(dataset[i][1])

	return dataset