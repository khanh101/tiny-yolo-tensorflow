create:
	python3 ./create_graph.py
train: create
	python3 ./train.py
clean:
	rm -r graph
	rm -r train_graph
