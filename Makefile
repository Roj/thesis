build:
	docker build . -t huergo
run:
	docker run --gpus all -v $(PWD)/graphs:/thesis/graphs -v $(PWD)/docker_logs:/thesis/logs -it huergo
