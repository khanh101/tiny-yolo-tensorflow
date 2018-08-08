create:
	python3 ./create_graph.py
train: graph
	python3 ./train.py
clean:
	rm -r graph
	rm -r train_graph
