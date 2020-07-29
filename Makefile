build:
	docker build . -t huergo
run:
	docker run --gpus all -v $(PWD)/graphs:/thesis/graphs -v $(PWD)/docker_logs:/thesis/logs -it huergo
run_calculin:
	docker run --runtime=nvidia -v $(PWD)/graphs:/thesis/graphs -v $(PWD)/experiments:/thesis/experiments -v $(PWD)/docker_logs:/thesis/logs -td huergo