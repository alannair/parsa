.ONESHELL:

.PHONY: parsa-image beam-image beam-prepare beam-image-0.23

vllm-beam:
	git clone --depth=1 --branch=beam/v0.11.0 https://gitlab-uk.rnd.huawei.com/sir_lab/power-aware-scheduling/vllm-beam.git

vllm-ascend-beam:
	git clone --depth=1 --branch=v0.11.0-beam https://gitlab-uk.rnd.huawei.com/sir_lab/power-aware-scheduling/vllm-ascend-beam.git

beam-image:
	python deploy.py build-image beam 

beam-prepare:
	python deploy.py prepare-vllm --version 0.23

# Requires the offline PyArrow 24.0.0 CPython 3.11 aarch64 wheel in ./wheelhouse.
beam-image-0.23:
	python deploy.py build-image beam --version 0.23

parsa-image:
	python deploy.py build-image parsa
