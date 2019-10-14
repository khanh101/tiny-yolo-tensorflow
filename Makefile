create:
	python3 ./create_graph.py
train: create
	python3 ./train.py
test:
	python3 ./test.py -i data/dog.jpg
clean:
	rm -r graph
	rm -r train_graph
