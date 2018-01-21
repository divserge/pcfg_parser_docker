This is a repository containing the Dockerfile for the image, which reproduces the experiments for my NLA application period project - PCFG parser. 

To build a new image from source and run
```bash
	sudo docker build -t parser_image_0 .
	sudo docker run -v absolute_path_to_local_dir:/parser/results parser_image_0
```

The image can also be found at:
https://cloud.docker.com/swarm/divserge/repository/docker/divserge/parser_image/general
