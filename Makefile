create:
	python3 -W ignore ./create_graph.py
train: create
	python3 -W ignore ./train.py
test:
	python3 ./test.py -i data/dog.jpg
clean:
	rm -r graph
	rm -r train_graph
