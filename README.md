This is a repository containing the Dockerfile for the image, which reproduces the experiments for my NLA application period project - PCFG parser. 

To build a new image from source and run
```bash
	sudo docker build -t parser_image_0 .
	sudo docker run -v absolute_path_to_local_dir:/parser/results parser_image_0
```

To pull the build image from the docker registry:
```bash
	sudo docker pull divserge/parser_image
```
